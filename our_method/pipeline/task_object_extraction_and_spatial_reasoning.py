import os
import re
import json
import cv2
import numpy as np
from pathlib import Path
from loguru import logger as log
import torch
from torchvision.ops import box_convert
import supervision as sv
from our_method.utils.processing_utils import prepare_output_dir


class TaskObjectExtractionAndSpatialReasoning:
    """
    Step 3 of the pipeline:
    - Load step_1 results
    - Annotate bounding boxes
    - GPT reasoning for task-relevant object extraction
    - Output scenario JSON files
    """

    # ---------------- Constants ----------------
    DO_NOT_MATCH_CATEGORIES = {"walls", "floors", "ceilings"}
    IMG_SHAPE_OG = (720, 1280)

    # ---------------- Init ----------------
    def __init__(self, gpt=None, verbose: bool = False):
        self.verbose = verbose
        self.gpt = gpt

        # internal variables
        self.save_dir = None
        self.step_1_output_info = None
        self.detected_categories_info = None

    # ---------------- Public API ----------------
    def __call__(
        self,
        step_1_output_path,
        step_2_output_path,
        step_3_output_path,
        goal_task=None,
        save_dir=None,
        visualize=False,
        mode="task",
        asset_pool=None,
        use_asset_pool_vocabulary=True,
        max_count_per_object=10,
        max_container_box_area=0.1,
    ):
        """
        High-level pipeline entry point.

        mode:
            "task"              기존 동작. goal_task 를 보고 필요한 새 물체를 상상해서
                                시나리오 여러 개를 만든다.
            "observed_contents" 입력 사진을 보고 이미 검출된 물체(그릇, 컵, 냄비 ...)
                                안에 실제로 보이는 내용물을 뽑는다. 어느 부모에 /
                                무엇을 / 몇 개 넣는지를 시나리오 하나로 낸다.
                                Step 1 은 테이블 위에 직접 놓인 물체만 검출하므로
                                그릇 속 내용물은 이 경로로 채운다.
        asset_pool: observed_contents 에서 내용물 이름을 retrieval 풀의 카테고리
            이름에 맞추라고 GPT 에 힌트로 준다 (use_asset_pool_vocabulary).
        max_count_per_object: GPT 가 센 개수의 상한.
        max_container_box_area: 정규화 bbox 면적이 이보다 큰 물체(테이블 등)는
            확대 크롭에서 뺀다. 부모 후보 목록에는 그대로 남는다.
        """
        if mode == "observed_contents":
            return self._run_observed_contents(
                step_1_output_path=step_1_output_path,
                goal_task=goal_task,
                save_dir=save_dir,
                asset_pool=asset_pool,
                use_asset_pool_vocabulary=use_asset_pool_vocabulary,
                max_count_per_object=max_count_per_object,
                max_container_box_area=max_container_box_area,
            )
        assert mode == "task", f"unknown mode: {mode}"

        # (Step 0) Setup
        self.save_dir = prepare_output_dir(step_1_output_path, save_dir, "task_object_extraction_and_spatial_reasoning")

        self._load_meta(step_1_output_path)

        if self.verbose:
            log.info("Object Extraction And Spatial Reasoning using GPT")

        # (Step 1) Annotate image
        annotated_image_path = self._create_annotated_image()

        # (Step 2) Query GPT
        gpt_resp = self._query_gpt(annotated_image_path, goal_task)
        if gpt_resp is None:
            return False, None

        # (Step 3) Extract scenario JSON
        scenario_json = self._extract_json(gpt_resp)
        if scenario_json is None:
            return False, None

        # (Step 4) Save scenario files
        scenario_paths = self._save_scenarios(scenario_json, goal_task)
        last_path = scenario_paths[-1] if scenario_paths else None

        if self.verbose:
            log.success("Completed Task Object Extraction And Spatial Reasoning!")

        return True, last_path

    # ---------------- Observed contents mode ----------------
    def _run_observed_contents(self, step_1_output_path, goal_task, save_dir, asset_pool,
                               use_asset_pool_vocabulary, max_count_per_object,
                               max_container_box_area):
        self.save_dir = prepare_output_dir(step_1_output_path, save_dir, "task_object_extraction_and_spatial_reasoning")
        self._load_meta(step_1_output_path)

        if self.verbose:
            log.info("Observed Contents Extraction using GPT (which parent / what / how many)")

        annotated_image_path = self._create_annotated_image()
        crops_path = self._create_container_crops(max_container_box_area)

        vocabulary = None
        if use_asset_pool_vocabulary and asset_pool is not None:
            from our_method.utils.asset_pool import resolve_asset_pool
            pool = resolve_asset_pool(asset_pool, verbose=False)
            vocabulary = sorted(pool.categories(replace_underscores=False))

        scene_objects = list(self.detected_categories_info["names"])
        payload = self._payload_observed_contents(
            raw_image_path=self.step_1_output_info["input_rgb"],
            annotated_image_path=annotated_image_path,
            crops_path=crops_path,
            scene_objects=scene_objects,
            goal_task=goal_task,
            vocabulary=vocabulary,
        )
        resp = self.gpt(payload=payload, verbose=self.verbose)
        log.info(f"GPT Response: {resp}")
        if resp is None:
            return False, None

        parsed = self._extract_json(resp)
        if parsed is None:
            return False, None

        objects = self._contents_to_objects(parsed, scene_objects, max_count_per_object)
        with open(os.path.join(self.save_dir, "observed_contents_gpt_response.txt"), "w") as f:
            f.write(resp)

        scenario_paths = self._save_scenarios({"scenario_0": {"objects": objects}}, goal_task)

        if self.verbose:
            for name, info in objects.items():
                log.info(f"  {info['count']} x {name} -> {info['placement']} {info['parent_object']}")
            log.success("Completed Observed Contents Extraction!")
        return True, scenario_paths[-1]

    def _create_container_crops(self, max_box_area, crop_size=384, pad_ratio=0.35):
        """검출된 물체마다 원본 사진을 확대 크롭해서 한 장으로 붙인다.

        전체 사진에서 그릇은 수십 픽셀이라 내용물이 잘 안 보이고, 어노테이션
        이미지는 라벨 글자가 내용물을 가린다. 크롭에는 박스를 그리지 않고 이름을
        위쪽 띠에만 적는다.
        """
        img = cv2.imread(self.step_1_output_info["input_rgb"])
        h, w = img.shape[:2]
        tiles = []
        for name, (cx, cy, bw, bh) in zip(self.detected_categories_info["names"],
                                          self.detected_categories_info["boxes"]):
            if bw * bh > max_box_area:
                continue
            side = max(bw * w, bh * h) * (1 + 2 * pad_ratio)
            x0 = int(max(0, cx * w - side / 2)); x1 = int(min(w, cx * w + side / 2))
            y0 = int(max(0, cy * h - side / 2)); y1 = int(min(h, cy * h + side / 2))
            crop = cv2.resize(img[y0:y1, x0:x1], (crop_size, crop_size), interpolation=cv2.INTER_CUBIC)
            strip = np.full((40, crop_size, 3), 255, dtype=np.uint8)
            cv2.putText(strip, name, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
            tiles.append(np.concatenate([strip, crop], axis=0))
        if not tiles:
            return None
        per_row = 4
        blank = np.zeros_like(tiles[0])
        rows = []
        for i in range(0, len(tiles), per_row):
            row = tiles[i:i + per_row]
            row += [blank] * (per_row - len(row))
            rows.append(np.concatenate(row, axis=1))
        out_path = os.path.join(self.save_dir, "container_crops.png")
        cv2.imwrite(out_path, np.concatenate(rows, axis=0))
        return out_path

    def _payload_observed_contents(self, raw_image_path, annotated_image_path, crops_path,
                                   scene_objects, goal_task, vocabulary):
        instructions = (
            "You are a professional simulation designer reconstructing a real scene in simulation.\n"
            "The scene objects listed below were already detected and placed in simulation. "
            "Objects that sit INSIDE another object (food in a bowl, items in a cup or pot, etc.) "
            "were NOT detected and are missing from the simulation.\n\n"
            "Your job: look carefully at the images and list every object that is visibly inside "
            "one of the scene objects. For each one give\n"
            "  - parent_object: the scene object it is inside (must be one of the scene object names exactly)\n"
            "  - name: a short snake_case name of the object itself (e.g. instant_noodle_block)\n"
            "  - count: how many separate pieces you can see in that parent (integer >= 1). "
            "Count individual pieces (e.g. 3 dumplings -> 3, 5 scallion slices -> 5).\n"
            "  - description: a few words about how it looks\n\n"
            "Rules:\n"
            "- Only report what is actually visible in the image. Do not invent objects the task might need.\n"
            "- Do not report the scene objects themselves, and do not report objects standing directly on a table.\n"
            "- If a content object is mentioned in the task, make sure it is included.\n"
            "- Empty containers get no entry.\n"
            "Answer with a one-sentence explanation followed by JSON only, in this format:\n"
            '{"contents": [{"parent_object": "bowl_0", "name": "apple", "count": 2, "description": "red apples"}]}'
        )
        user_prompt = f"Task: {goal_task}\nScene objects: {scene_objects}\n"
        if vocabulary:
            user_prompt += (
                "Available simulation asset categories (prefer one of these names when it matches "
                f"what you see, otherwise use your own name): {vocabulary}\n"
            )
        user_prompt += (
            "Image 1: the original photo. Image 2: the same photo with detected scene objects labeled. "
            "Image 3 (if present): zoomed crops of each small scene object, labeled with its name above the crop."
        )
        content = [{"type": "input_text", "text": user_prompt}]
        for path in (raw_image_path, annotated_image_path, crops_path):
            if path is None:
                continue
            ext = os.path.splitext(path)[1].lstrip(".").lower() or "png"
            content.append({
                "type": "input_image",
                "image_url": f"data:image/{ext};base64,{self.gpt.encode_image(path)}",
                "detail": "high",
            })
        return {
            "model": self.gpt.VERSIONS[self.gpt.version],
            "instructions": instructions,
            "input": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_output_tokens": 2000,
        }

    @staticmethod
    def _contents_to_objects(parsed, scene_objects, max_count):
        """GPT 의 contents 리스트를 Step 6 이 읽는 objects dict 로 바꾼다.

        키는 내용물 이름. 같은 이름이 다른 부모에 또 나오면 _1, _2 를 붙인다.
        parent_object 가 씬에 없으면 버린다.
        """
        items = parsed.get("contents", []) if isinstance(parsed, dict) else []
        objects = {}
        for it in items:
            parent = str(it.get("parent_object", "")).strip()
            if parent not in scene_objects:
                log.warning(f"observed_contents: unknown parent '{parent}' -> skip {it}")
                continue
            name = re.sub(r"[^a-z0-9_]+", "_", str(it.get("name", "")).strip().lower()).strip("_")
            if not name or name in scene_objects:
                log.warning(f"observed_contents: bad name -> skip {it}")
                continue
            try:
                count = int(it.get("count", 1))
            except (TypeError, ValueError):
                count = 1
            count = max(1, min(count, max_count))
            key, k = name, 1
            while key in objects:
                key = f"{name}_{k}"; k += 1
            objects[key] = {
                "parent_object": parent,
                "placement": "inside",
                "count": count,
                "description": str(it.get("description", "")),
            }
        return objects

    # ---------------- Internal: Setup ----------------
    def _load_meta(self, step_1_output_path):
        with open(step_1_output_path, "r") as f:
            self.step_1_output_info = json.load(f)

        # load detection results
        det_path = self.step_1_output_info["detected_categories"]
        with open(det_path, "r") as f:
            self.detected_categories_info = json.load(f)

    # ---------------- Internal: Annotation ----------------
    def _create_annotated_image(self):
        img_path = self.step_1_output_info["input_rgb"]
        detected_info = self.detected_categories_info

        annotated = self._draw_bboxes(img_path, detected_info)
        out_path = os.path.join(self.save_dir, "annotated_image.png")
        cv2.imwrite(out_path, annotated)

        if self.verbose:
            log.info(f"Annotated image saved to: {out_path}")

        return out_path

    def _draw_bboxes(self, img_path, detected_info):
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        names = detected_info["names"]
        boxes = torch.tensor(detected_info["boxes"])
        logits = torch.tensor(detected_info["logits"])

        boxes_pixel = boxes * torch.tensor([w, h, w, h])
        xyxy = box_convert(boxes_pixel, in_fmt="cxcywh", out_fmt="xyxy").numpy().astype(np.int16)

        labels = [f"{name} {score:.2f}" for name, score in zip(names, logits)]

        detections = sv.Detections(xyxy=xyxy)
        box_annot = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
        label_annot = sv.LabelAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            text_position=sv.geometry.core.Position.CENTER,
            text_padding=2,
        )

        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_bgr = box_annot.annotate(img_bgr, detections)
        img_bgr = label_annot.annotate(img_bgr, detections, labels)

        return img_bgr

    # ---------------- Internal: GPT ----------------
    def _query_gpt(self, annotated_image_path, goal_task):
        if self.verbose:
            log.debug("Query GPT for task object reasoning...")

        scene_objects = str(self.detected_categories_info["names"])

        payload = self.gpt.payload_task_object_extraction_and_spatial_reasoning(
            annotated_image_path=annotated_image_path,
            scene_objects=scene_objects,
            goal_task=goal_task,
        )

        resp = self.gpt(payload=payload, verbose=self.verbose)
        log.info(f"GPT Response: {resp}")

        return resp

    # ---------------- Internal: JSON Extraction ----------------
    def _extract_json(self, text):
        """GPT 응답에서 JSON 을 꺼낸다.

        ```json 펜스를 붙여 주면 그걸 쓰고, 없으면 본문에서 중괄호 균형을 세어
        가장 바깥 객체를 찾는다. GPT 가 설명 문장을 앞에 붙이고 펜스 없이 생
        JSON 을 뱉는 경우가 있어서(task 에 개수를 넣으면 자주 그런다) 펜스만
        받으면 멀쩡한 응답을 통째로 버리게 된다.
        """
        for pattern in (r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```"):
            m = re.search(pattern, text)
            if m:
                try:
                    return json.loads(m.group(1))
                except json.JSONDecodeError as e:
                    log.error(f"JSON parsing failed: {e}")
                    return None

        # 펜스가 없다 -> 중괄호 균형으로 가장 바깥 JSON 객체를 잘라낸다
        start = text.find("{")
        while start != -1:
            depth, in_str, esc = 0, False, False
            for i in range(start, len(text)):
                c = text[i]
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                    continue
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except json.JSONDecodeError:
                            break
            start = text.find("{", start + 1)

        log.error("No JSON found inside GPT response.")
        return None

    # ---------------- Internal: Saving ----------------
    def _save_scenarios(self, scenario_json, goal_task):
        scenario_paths = []

        for i, (_, content) in enumerate(scenario_json.items()):
            out_path = os.path.join(self.save_dir, f"task_obj_output_info_scenario_{i}.json")
            payload = {
                "task": goal_task,
                "objects": content.get("objects", []),
            }
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=4)

            scenario_paths.append(out_path)

        # index file
        index_path = os.path.join(self.save_dir, "task_obj_output_info.json")
        with open(index_path, "w") as f:
            json.dump(scenario_paths, f, indent=4)

        return scenario_paths
