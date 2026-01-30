import enum
from sklearn import feature_extraction
import torch
from torchvision.ops.boxes import _box_xyxy_to_cxcywh
from groundingdino.util.inference import load_image
import numpy as np
from pathlib import Path
from PIL import Image
import datetime
import os
import copy
import json
import cv2
import faiss
import digital_cousins
import re
import shutil
from itertools import product
import warnings
import supervision as sv
from torchvision.ops import box_convert
import our_method
import our_method.utils.transform_utils as T
from our_method.models.clip import CLIPEncoder
from our_method.models.gpt import GPT
from our_method.utils.processing_utils import NumpyTorchEncoder, compute_bbox_from_mask
from our_method.utils.dataset_utils import get_all_dataset_categories, get_all_articulated_categories, \
    extract_info_from_model_snapshot, ARTICULATION_INFO, ARTICULATION_VALID_ANGLES


DO_NOT_MATCH_CATEGORIES = {"walls", "floors", "ceilings"}
IMG_SHAPE_OG = (720, 1280)

class TaskObjectExtractionAndSpatialReasoning:
    """
    

    Inputs:
        - 

    Outputs:
        - 
    """

    def __init__(
            self,
            verbose=False,
    ):
        """
        Args:
            verbose (bool): Whether to display verbose print outs during execution or not
        """
        self.verbose = verbose

    def __call__(
            self,
            step_1_output_path,
            step_2_output_path,
            step_3_output_path,
            gpt_api_key,
            gpt_version="4o",
            goal_task = None,
            save_dir=None
    ):
        """
        

        Args:
            

        Returns:
            
        """
        # Sanity check values
        # Parse save_dir, and create the directory if it doesn't exist
        if save_dir is None:
            save_dir = os.path.dirname(os.path.dirname(step_1_output_path))
        save_dir = os.path.join(save_dir, "task_object_extraction_and_spatial_reasoning")
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Load meta info
        with open(step_1_output_path, "r") as f:
            step_1_output_info = json.load(f)

        # scene objects
        with open(step_1_output_info["detected_categories"], "r") as f:
            detected_categories_info = json.load(f)

        # annotated image
        real_scene_img_path = step_1_output_info["input_rgb"]
        annotated_image = self.image_bbox_annotations(real_scene_img_path, detected_categories_info)
        annotated_image_path = os.path.join(save_dir, "annotated_image.png")
        cv2.imwrite(annotated_image_path, annotated_image)

        # GPT
        if self.verbose:
            print("""

##################################################################
### Object Extraction And Spatial Reasoning using GPT ###
##################################################################

            """)
        gpt = GPT(api_key=gpt_api_key, version=gpt_version, log_dir_tail="_TaskObjExtractionAndSpatialReasoning")

        scene_objects_str = str(detected_categories_info["names"])
        # print(f"scene_objects_str: {scene_objects_str}")

        # get GPT query payload
        task_object_extraction_and_spatial_reasoning_payload = gpt.payload_task_object_extraction_and_spatial_reasoning(
            annotated_image_path=annotated_image_path,
            scene_objects=scene_objects_str,
            goal_task=goal_task,
        )

        # query GPT
        gpt_text_response = gpt(task_object_extraction_and_spatial_reasoning_payload)
        print("GPT Response:\n", gpt_text_response)

        if gpt_text_response is None:
            # Failed, terminate early
            return False, None

        # Parse GPT response
        gpt_json = self.extract_json_from_text(gpt_text_response)
        print(f"gpt_json: {gpt_json}")

        # Split scenarios
        task_object_extraction_and_spatial_reasoning_info_list = []
        for scenario_num, scenario_info in gpt_json.items():
            # print(f"scenario_num: {scenario_num}")
            # print(f"scenario_info: {scenario_info}")
            task_object_extraction_and_spatial_reasoning_info_scenario = {
                "task": goal_task,
                "objects": scenario_info["objects"]
            }
            task_object_extraction_and_spatial_reasoning_info_list.append(task_object_extraction_and_spatial_reasoning_info_scenario)

        # Write Scenario files
        task_output_info_path = []
        for i, task_object_extraction_and_spatial_reasoning_info in enumerate(task_object_extraction_and_spatial_reasoning_info_list):
            task_object_extraction_and_spatial_reasoning_path = f"{save_dir}/task_obj_output_info_scenario_{i}.json"
            task_output_info_path.append(task_object_extraction_and_spatial_reasoning_path)
            with open(task_object_extraction_and_spatial_reasoning_path, "w+") as f:
                json.dump(task_object_extraction_and_spatial_reasoning_info, f, indent=4)

        # Write a path file
        with open(f"{save_dir}/task_obj_output_info.json", "w+") as f:
            json.dump(task_output_info_path, f, indent=4)

        print("""

###############################################################
### Completed Task Object Extraction And Spatial Reasoning! ###
###############################################################

        """)
        return True, task_object_extraction_and_spatial_reasoning_path

    def extract_json_from_text(self, text):
        """
        Extract JSON data from text that contains triple backtick delimited JSON.
        """
        # Regex pattern to find text between ```json and ```
        pattern = r'```json\s*([\s\S]*?)\s*```'

        # Search for the pattern in the text
        match = re.search(pattern, text)

        if match:
            # Extract the JSON string
            json_str = match.group(1)

            try:
                # Parse the JSON string
                json_data = json.loads(json_str)
                return json_data
            except json.JSONDecodeError as e:
                print(f"❌ Error parsing JSON: {e}")
                return None
        else:
            print("No JSON found in the text")
            return None

    def image_bbox_annotations(self, real_scene_img_path, detected_categories_info):
        """
        Loads the image and bounding box annotations, annotates them using supervision,
        and returns the annotated image + bounding box info.

        Args:
            real_scene_img_path (str): Path to real scene image.
            detected_categories_info (dict): Dictionary containing object names, boxes, logits, etc.

        Returns:
            annotated_image (np.ndarray): Annotated image (BGR).
            xyxy_boxes (np.ndarray): Bounding boxes in (x1, y1, x2, y2) format.
        """

        # Load and prepare image
        test_image = cv2.cvtColor(cv2.imread(real_scene_img_path), cv2.COLOR_BGR2RGB)
        h, w, _ = test_image.shape

        # Load detection info
        names = detected_categories_info['names']
        boxes = torch.tensor(detected_categories_info['boxes'])  # cxcywh format, normalized
        logits = torch.tensor(detected_categories_info['logits'])

        # Convert boxes to pixel values and xyxy format
        boxes_pixel = boxes * torch.tensor([w, h, w, h])
        xyxy_boxes = box_convert(boxes=boxes_pixel, in_fmt='cxcywh', out_fmt='xyxy').numpy().astype(np.int16)

        # Labels: name + score
        labels = [f"{name} {score:.2f}" for name, score in zip(names, logits)]

        # Supervision annotation
        detections = sv.Detections(xyxy=xyxy_boxes)
        bbox_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
        label_annotator = sv.LabelAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            text_position=sv.geometry.core.Position.CENTER,
            text_padding=2
        )

        # Annotate
        annotated_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)
        annotated_image = bbox_annotator.annotate(scene=annotated_image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        return annotated_image
