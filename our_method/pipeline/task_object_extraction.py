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
import our_method
import our_method.utils.transform_utils as T
from our_method.models.clip import CLIPEncoder
from our_method.models.gpt import GPT
from our_method.utils.processing_utils import NumpyTorchEncoder, compute_bbox_from_mask
from our_method.utils.dataset_utils import get_all_dataset_categories, get_all_articulated_categories, \
    extract_info_from_model_snapshot, ARTICULATION_INFO, ARTICULATION_VALID_ANGLES


DO_NOT_MATCH_CATEGORIES = {"walls", "floors", "ceilings"}
IMG_SHAPE_OG = (720, 1280)

class TaskObjectExtraction:
    """
    2nd Step in ACDC pipeline. This takes in the output from Step 1 (Real World Extraction) and generates
    ordered digital cousin candidates from a given dataset (default is Behavior-1K dataset)

    Foundation models used:
        - GPT-4O (https://openai.com/index/hello-gpt-4o/)
        - CLIP (https://github.com/openai/CLIP)
        - DINOv2 (https://github.com/facebookresearch/dinov2)

    Inputs:
        - Output from Step 1, which includes the following:
            - Camera Intrinsics Matrix
            - Detected Categories information
            - Floor segmentation mask
            - Wall(s) segmentation mask(s)
            - Estimated z-direction in the camera frame
            - Selected origin position in the camera frame
            - Input RGB image
            - Input (linear) Depth image (potentially synthetically generated)
            - Depth limits (min, max)
            - Mount type

    Outputs:
        - Ordered digital cousin (category, model, pose) information per detected object from Step 1
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
        Runs the digital cousin matcher. This does the following steps for each detected object from Step 1:

        1. Use CLIP embeddings to find the top-K nearest OmniGibson dataset categories for a given box + mask
        2. Select digital cousins using encoder features + GPT

        Args:
            step_1_output_path (str): Absolute path to the output file generated from Step 1 (RealWorldExtractor)
            gpt_api_key (str): Valid GPT-4O compatible API key
            gpt_version (str): GPT version to use. Valid options are {"4o", "4v"}.
                Default is "4o", which we've found to work empirically better than 4V
            top_k_categories (int): Number of closest categories from the OmniGibson dataset from which digital
                cousin candidates will be selected
            top_k_models (int): Number of closest asset digital cousin models from the OmniGibson dataset to select
            top_k_poses (int): Number of closest asset digital cousin model poses to select
            n_digital_cousins (int): Number of digital cousins to select. This number cannot be greater than
                @top_k_models
            n_cousins_reselect_cand (int): The frequency of reselecting digital cousin candidates.
                If set to 1, reselect candidates for each digital cousin.
            remove_background (bool): Whether to remove background before computing visual encoder features when
                computing digital cousin candidates
            gpt_select_cousins (bool): Whether to prompt GPT to select nearest asset as a digital cousin.
                If False, the nearest digital cousin will be the one with the least DINOv2 embedding distance.
            start_at_name (None or str): If specified, the name of the object to start at. This is useful in case
                the pipeline crashes midway, and can resume progress without starting from the beginning
            n_cousins_link_count_threshold (int): The number of digital cousins to apply door/drawer count threshold
                during selection. When selecting digital cousin candidates for articulated objects, setting this as a
                positive integer will leverage the GPT-driven door / drawer annotations from Step 1 to further constrain
                the potential candidates during visual encoder selection.
                If set to 0, this threshold will not be used.
                If set larger than n_digital_cousins, this threshold will always be used.
            save_dir (None or str): If specified, the absolute path to the directory to save all generated outputs. If
                not specified, will generate a new directory in the same directory as @step_1_output_path

        Returns:
            2-tuple:
                bool: True if the process completed successfully. If successful, this will write all relevant outputs to
                    the directory specified in the second output
                None or str: If successful, this will be the absolute path to the main output file. Otherwise, None
        """
        # Sanity check values
        # Parse save_dir, and create the directory if it doesn't exist
        if save_dir is None:
            save_dir = os.path.dirname(os.path.dirname(step_1_output_path))
        save_dir = os.path.join(save_dir, "task_object_extraction")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        if self.verbose:
            print(f"Computing digital cousins given output {step_1_output_path}...")

        if self.verbose:
            print("""

##################################################################
### 1. Use CLIP embeddings to find top-K categories per object ###
##################################################################

            """)

        # Load meta info
        with open(step_1_output_path, "r") as f:
            step_1_output_info = json.load(f)
        
        with open(step_1_output_info["detected_categories"], "r") as f:
            detected_categories_info = json.load(f)
        
        gpt = GPT(api_key=gpt_api_key, version=gpt_version)

        scene_objects_str = str(set(detected_categories_info["phrases"])).strip('[]')
        # task = "Open the cabinet next to the locker and give me the cup inside"
        # task = "Give me the cup in the cabinet above the microwave"
        # task = "Give me the orange in the cabinet above the microwave"

        nn_selection_payload = gpt.payload_task_object_extraction(
            scene_objects_str,
            goal_task=goal_task,
        )

        gpt_text_response = gpt(nn_selection_payload)

        if gpt_text_response is None:
            # Failed, terminate early
            return False, None

        if "json" in gpt_text_response.lower() and "```" in gpt_text_response:
            # ```json 또는 ```으로 감싸진 블록 제거
            gpt_text_response = re.sub(r"^```[a-z]*\n|\n```$", "", gpt_text_response.strip(), flags=re.IGNORECASE)

        try:
            gpt_json = json.loads(gpt_text_response)
        except json.JSONDecodeError as e:
            print("❌ JSON 파싱 실패:", e)
            return False, None
        

        task_object_extraction_info = {
            "task": goal_task,
            "objects": gpt_json
        }
        
        task_object_extraction_path = f"{save_dir}/target_object_extraction.json"
    
        with open(task_object_extraction_path, "w+") as f:
            json.dump(task_object_extraction_info, f, indent=4)

        print("""

##########################################
### Completed Task Object Extraction! ###
##########################################

        """)
        return True, task_object_extraction_path