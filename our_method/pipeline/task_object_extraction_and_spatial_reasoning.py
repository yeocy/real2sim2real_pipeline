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
    ):
        """
        High-level pipeline entry point.
        """

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
        pattern = r"```json\s*([\s\S]*?)\s*```"
        m = re.search(pattern, text)

        if not m:
            log.error("No JSON found inside GPT response.")
            return None

        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError as e:
            log.error(f"JSON parsing failed: {e}")
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
