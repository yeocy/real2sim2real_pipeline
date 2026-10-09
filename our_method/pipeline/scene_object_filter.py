"""
scene_object_filter.py — 트윈 씬에 넣지 않을 검출을 GPT 로 고른다.

두 위치에서 쓴다.
  - Step 1 직후 (SAM3D 생성 모드): 매칭 정보 없이 판단한다. keep 된 물체만 SAM3D 로 만든다.
  - Step 2 뒤, Step 3 전 (기존 풀 모드): 매칭된 에셋까지 보고 판단한다 (규칙 D4).

Step 1 은 사진의 모든 물체를 잡는다. 배경(벽걸이 TV), 같은 물체를 쪼갠 중복 박스, 풀에 맞는 에셋이
없어 엉뚱한 카테고리로 매칭된 배경 물체까지 Step 3 에 그대로 배치되면 씬이 망가진다
(예: LIBERO-10 task 04 의 television_0/1 -> breakfast_table). 지금까지는 config 의 discard_objs 에
이름을 손으로 적었는데, 이름이 실행마다 달라서 다른 태스크에 쓸 수 없었다.

여기서는 고정된 규칙 프롬프트(GPT.payload_filter_scene_objects)로 물체마다 keep / discard 를 받는다.
GPT 에게는 다음을 준다.
  - 입력 사진에 검출 이름 박스를 그린 이미지
  - 검출마다 크롭 + Step 2 가 고른 에셋 스냅샷 (매칭이 틀렸는지 판단 근거)
  - 태스크 문장 (태스크에 나오는 물체는 항상 keep)

결과는 <save_dir>/<out_subdir>/object_filter.json 에 남기고, discard 이름 목록을 돌려준다.
GAIA.run 이 이 목록을 RealSceneGenerator 의 discard_objs 에 합친다.
작업면 가구(테이블 등)는 규칙 D0 으로 항상 뺀다 (Step 1 이 그 윗면을 씬 바닥으로 쓰기 때문).
"""
import json
import os
import re

import cv2
from loguru import logger as log


class SceneObjectFilter:
    def __init__(self, gpt, verbose=True):
        self.gpt = gpt
        self.verbose = verbose

    @staticmethod
    def _annotate(img_path, names, boxes, out_path):
        """검출 박스 + 이름. boxes 는 정규화 (cx, cy, w, h)."""
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        for name, (cx, cy, bw, bh) in zip(names, boxes):
            x0, y0 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x1, y1 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
            (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ty = max(y0 - 6, th + 4)
            cv2.rectangle(img, (x0, ty - th - 4), (x0 + tw + 6, ty + 4), (0, 0, 0), -1)
            cv2.putText(img, name, (x0 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(out_path, img)

    @staticmethod
    def _parse(text, names):
        m = re.search(r"\{.*\}", text or "", re.S)
        if m is None:
            raise ValueError(f"GPT 응답에서 JSON 을 찾지 못했다: {text!r}")
        rows = {r["name"]: r for r in json.loads(m.group(0)).get("objects", []) if r.get("name") in names}
        missing = [n for n in names if n not in rows]
        if missing:
            # 판단이 빠진 검출은 지우지 않는다 (잘못 빼는 것보다 남기는 쪽이 안전)
            log.warning(f"[SceneObjectFilter] GPT 판단이 빠진 검출은 keep: {missing}")
            for n in missing:
                rows[n] = {"name": n, "decision": "keep", "rule": "-", "reason": "no decision from GPT"}
        return [rows[n] for n in names]

    def __call__(self, step_1_output_path, step_2_output_path=None, goal_task=None, save_dir=None,
                 enabled=True, use_cache=True, out_subdir=None):
        """
        Args:
            step_2_output_path (None or str): 주면 매칭된 에셋도 GPT 에게 보여준다 (Step 2 뒤에서 쓸 때)

        Returns:
            list of str: discard 할 검출 이름
        """
        if not enabled:
            return []
        step_1 = json.load(open(step_1_output_path))
        det = json.load(open(step_1["detected_categories"]))
        step_2 = json.load(open(step_2_output_path))["objects"] if step_2_output_path else None
        save_dir = save_dir or os.path.dirname(os.path.dirname(step_1_output_path))
        out_subdir = out_subdir or ("step_2_5_output" if step_2 is not None else "step_1_5_output")
        out_dir = os.path.join(save_dir, out_subdir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "object_filter.json")

        names = [n for n in det["names"] if step_2 is None or n in step_2]
        if use_cache and os.path.exists(out_path):
            cached = json.load(open(out_path))
            if sorted(cached.get("names", [])) == sorted(names) and cached.get("goal_task") == goal_task:
                log.info(f"[SceneObjectFilter] 이전 결과 사용: {out_path}")
                return cached["discard"]

        annotated = os.path.join(out_dir, "detections_annotated.png")
        idx = {n: i for i, n in enumerate(det["names"])}
        self._annotate(step_1["input_rgb"], names, [det["boxes"][idx[n]] for n in names], annotated)

        seg_dir = det["segmentation_dir"]
        objects = []
        for n in names:
            o = {"name": n, "phrase": det["phrases"][idx[n]],
                 "crop_path": os.path.join(seg_dir, f"{n}_nonprojected_crop.png")}
            if step_2 is not None:
                top = step_2[n]["cousins"][0]
                o.update(match_category=top["category"], match_model=top["model"],
                         snapshot_path=top.get("snapshot"))
            objects.append(o)

        text = self.gpt(self.gpt.payload_filter_scene_objects(annotated, objects, goal_task),
                        verbose=self.verbose)
        rows = self._parse(text, names)
        discard = [r["name"] for r in rows if r.get("decision") == "discard"]

        json.dump({"goal_task": goal_task, "names": names, "decisions": rows, "discard": discard,
                   "keep": [n for n in names if n not in discard], "annotated_image": annotated},
                  open(out_path, "w"), indent=2)
        for r in rows:
            log.info(f"[SceneObjectFilter] {r['name']:16s} {r.get('decision', '?'):8s} "
                     f"{r.get('rule', '')}: {r.get('reason', '')}")
        log.info(f"[SceneObjectFilter] discard {discard} -> {out_path}")
        return discard
