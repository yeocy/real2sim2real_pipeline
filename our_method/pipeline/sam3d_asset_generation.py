"""
sam3d_asset_generation.py — Step 1.5: Step 1 이 검출하고 필터(GPT)가 남긴 물체를 SAM3D 로 만들어 풀을 짓는다.

씬에 넣을 물체를 사람이 고르지 않는다. 고르는 것은 모두 API(GPT) 다.
  Step 1  (GPT)  물체 캡션 + 마스크
  필터    (GPT)  SceneObjectFilter: keep / discard (작업면 테이블은 D0 으로 항상 뺀다)
  여기서         keep 된 검출마다 Step 1 마스크로 SAM3D 메시 생성 -> sam3d_asset_builder.py 로 풀 생성
                 (실치수 굽기, 텍스처, OG USD, retrieval 뷰 렌더, overview.png)
                 + [align] 입력 카메라 render-and-compare 로 yaw(z_angle)와 바닥 xy 를 맞춘다
  Step 2/3       그 풀로 매칭, 배치. 풀 모델은 실치수가 구워져 있어 Step 3 가 스케일을 바꾸지 않는다.
                 정렬 pose(registration.pose_cam)가 있는 검출은 Step 3 이 Step 2 의 cousin/yaw 대신
                 자기 에셋을 그 pose 로 놓는다 (asset_pool.sam3d_pose, real_scene_generation).

이름 규칙 (사람이 적지 않는다):
  category = Step 1 캡션을 소문자 + '_' 로 (예: "small plate" -> small_plate)
  model    = 검출 이름에서 '_' 를 뺀 것 (예: small_plate_1 -> smallplate1). 모델명에 '_' 는 쓸 수 없다.
"""
import os
import re
import sys
import json
import shutil
import subprocess

import cv2
import numpy as np
import yaml
from loguru import logger as log

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUILDER = os.path.join(REPO, "our_method_test", "sam3d_asset_builder.py")
IMAGE_STEM = "scene"


def generator_call(config):
    return config["pipeline"].get("SAM3DAssetGenerator", {}).get("call", {})


def sam3d_enabled(config):
    return bool(generator_call(config).get("enabled", False))


def sam3d_pool_root(config, save_dir):
    """생성 풀 위치. call.pool_root 가 없으면 <save_dir>/sam3d_pool (결과 폴더 안에 같이 남는다)."""
    root = generator_call(config).get("pool_root")
    return os.path.abspath(root) if root else os.path.join(os.path.abspath(save_dir), "sam3d_pool")


def _category(phrase):
    return re.sub(r"[^a-z0-9]+", "_", phrase.lower()).strip("_")


def _model(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


class SAM3DAssetGenerator:
    def __call__(self, step_1_output_path, keep_names, save_dir, inputs_dir, pool_root,
                 image=None, gpus=None, elevation=40.0, title=None, use_cache=True):
        """
        Args:
            keep_names (list of str): 필터가 keep 한 Step 1 검출 이름
            inputs_dir (str): camera_depth.npy, camera_info.json 이 있는 입력 폴더 (실치수 측정용)
            image (None or str): SAM3D 입력 이미지. None 이면 Step 1 입력(input_rgb).
                같은 카메라의 고해상도 이미지가 있으면 주는 게 좋다 (작은 물체 품질)
            gpus (None or str): 원격 SAM3D 서버의 빈 GPU (예: "6,7")

        Returns:
            str: 풀 루트
        """
        step_1 = json.load(open(step_1_output_path))
        det = json.load(open(step_1["detected_categories"]))
        idx = {n: i for i, n in enumerate(det["names"])}
        assets = [{"instance": f"{n}_0", "detection": n, "category": _category(det["phrases"][idx[n]]),
                   "model": _model(n), "mass": 0.3} for n in keep_names]

        report = os.path.join(pool_root, "sam3d_assets.json")
        sam3d_dir = os.path.join(os.path.abspath(save_dir), "sam3d")
        spec_path = os.path.join(sam3d_dir, "spec.yaml")
        if use_cache and os.path.exists(report):
            rows = json.load(open(report))
            have = {(r["category"], r["model"]) for r in rows}
            if have == {(a["category"], a["model"]) for a in assets} and all(
                    os.path.exists(os.path.join(pool_root, "og_dataset", "objects", a["category"], a["model"]))
                    for a in assets):
                log.info(f"[SAM3D] 같은 물체의 풀이 이미 있다 -> 재사용: {pool_root}")
                from our_method.utils.sam3d_pose_alignment import ALIGN_METHOD
                stale = [r["model"] for r in rows if (r.get("registration") or {}).get("method") != ALIGN_METHOD]
                if stale and os.path.exists(spec_path):
                    # 정렬(align) 전이나 이전 방식으로 만든 풀: 정렬 후 크기가 바뀌므로 USD, 렌더까지 다시 만든다
                    log.info(f"[SAM3D] 정렬({ALIGN_METHOD}) 결과가 없다 {stale} -> align,usd,render,overview 실행")
                    self._run_builder(spec_path, stages="align,usd,render,overview")
                return pool_root

        mask_dir = os.path.join(sam3d_dir, "sam3", IMAGE_STEM)
        os.makedirs(mask_dir, exist_ok=True)
        src = image or step_1["input_rgb"]
        img_path = os.path.join(sam3d_dir, f"{IMAGE_STEM}.png")
        img = cv2.imread(src)
        cv2.imwrite(img_path, img)
        h, w = img.shape[:2]

        # Step 1 마스크(입력 해상도) -> SAM3D 이미지 해상도, 빌더 레이아웃 sam3/<stem>/<prompt>.npy (1,H,W)
        # *_mask_pruned.png 는 점군용으로 픽셀을 솎아낸 점무늬라 SAM3D 입력/치수 측정에 쓰면 안 된다.
        seg_dir = det["segmentation_dir"]
        for n in keep_names:
            m = cv2.imread(os.path.join(seg_dir, f"{n}_nonprojected_mask.png"), cv2.IMREAD_GRAYSCALE)
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 127
            np.save(os.path.join(mask_dir, f"{n}.npy"), m[None])

        spec = {
            "pool": pool_root,
            "inputs": os.path.abspath(inputs_dir),
            "sam3d_dir": sam3d_dir,
            "image_stem": IMAGE_STEM,
            "title": title or "",
            "elevation": float(elevation),
            "sam3d": {"image": img_path, "from_masks": True, "gpus": gpus},
            "link_categories": [],
            "assets": assets,
        }
        with open(spec_path, "w") as f:
            yaml.safe_dump(spec, f, allow_unicode=True, sort_keys=False)
        log.info(f"[SAM3D] {len(assets)}개 물체 -> {[a['category'] + '/' + a['model'] for a in assets]}")
        log.info(f"[SAM3D] spec: {spec_path}")

        ret = self._run_builder(spec_path)
        # Isaac 은 종료 시 segfault(139) 를 내기도 한다. 산출물로 성공 여부를 본다.
        missing = [a for a in assets if not os.path.exists(os.path.join(
            pool_root, "og_dataset", "objects", a["category"], a["model"], "usd",
            f"{a['model']}.encrypted.usd"))]
        if missing or not os.path.exists(report):
            raise RuntimeError(f"[SAM3D] 풀 생성 실패 (exit {ret}), 빠진 모델: {[a['model'] for a in missing]}")
        log.info(f"[SAM3D] 풀 생성 완료: {pool_root}  (overview: {os.path.join(pool_root, 'overview.png')}, "
                 f"yaw 정렬: {os.path.join(pool_root, 'alignment.png')})")
        return pool_root

    @staticmethod
    def _run_builder(spec_path, stages=None):
        """빌더는 OmniGibson 을 띄우므로 별도 프로세스로 돌린다 (이 프로세스의 sim 상태와 섞이지 않게)."""
        env = dict(os.environ)
        env["PYTHONPATH"] = REPO + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        cmd = [sys.executable, BUILDER, "--spec", spec_path] + (["--stages", stages] if stages else [])
        return subprocess.run(cmd, env=env).returncode
