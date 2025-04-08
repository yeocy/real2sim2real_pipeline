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
import omnigibson as og
from omnigibson.objects import DatasetObject
from omnigibson.scenes import Scene
import our_method
import our_method.utils.transform_utils as T
from our_method.models.clip import CLIPEncoder
from our_method.models.gpt import GPT
from our_method.utils.processing_utils import NumpyTorchEncoder, compute_bbox_from_mask
from our_method.utils.dataset_utils import get_all_dataset_categories, get_all_articulated_categories, \
    extract_info_from_model_snapshot, ARTICULATION_INFO, ARTICULATION_VALID_ANGLES
from our_method.utils.scene_utils import compute_obj_bbox_info
from our_method.pipeline.task_scene_generation import TaskSceneGenerator

DO_NOT_MATCH_CATEGORIES = {"walls", "floors", "ceilings"}
IMG_SHAPE_OG = (720, 1280)

class TaskObjectResizing:
    """
    Description

    Inputs:
        

    Outputs:
        
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
            task_feature_matching_path,
            gpt_api_key,
            gpt_version="4o",
            save_dir=None
    ):
        # Parse save_dir, and create the directory if it doesn't exist
        if save_dir is None:
            save_dir = os.path.dirname(os.path.dirname(task_feature_matching_path))
        save_dir = os.path.join(save_dir, "task_object_resizing")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        # Load meta info
        with open(task_feature_matching_path, "r") as f:
            task_object_retrieval_output_info = json.load(f)


        # Launch omnigibson
        og.launch()
        og.sim.stop()
        og.clear()
        scene = Scene(use_floor_plane=True, floor_plane_visible=True, use_skybox=True)
        og.sim.import_scene(scene)
        og.sim.play()



        gpt = GPT(api_key=gpt_api_key, version=gpt_version, log_dir_tail="_TaskObjResizing")
        json_list = []
        for scenario_obj_num_json_path in task_object_retrieval_output_info:
            with open(scenario_obj_num_json_path, "r") as f:
                task_obj_output_info = json.load(f)

            # Get current object size
            obj_size_info = {}
            for obj_name, obj_info in task_obj_output_info["objects"].items():
                obj = DatasetObject(
                        name=obj_name,
                        category=obj_info["category"],
                        model=obj_info["model"],
                        visual_only=True,
                        scale=[1,1,1]
                    )
                scene.add_object(obj)  # Add the object in the scene
                og.sim.step()

                # Get Object Bounding Box
                obj_bbox_info = compute_obj_bbox_info(obj=obj)
                bbox_bottom_in_desired_frame = obj_bbox_info['bbox_bottom_in_desired_frame']
                bbox_top_in_desired_frame = obj_bbox_info['bbox_top_in_desired_frame']

                # Calculate dimensions of the object
                x_dim = np.linalg.norm(bbox_bottom_in_desired_frame[0] - bbox_bottom_in_desired_frame[1])
                y_dim = np.linalg.norm(bbox_bottom_in_desired_frame[3] - bbox_bottom_in_desired_frame[0])
                z_dim = np.linalg.norm(bbox_bottom_in_desired_frame[0] - bbox_top_in_desired_frame[0])
                # print(f"[{x_dim}, {y_dim}, {z_dim}]")

                obj_size_info[obj_name] = [x_dim, y_dim, z_dim]

                scene.remove_object(obj)

            # Setup GPT prompt
            task_object_resizing_payload = gpt.payload_task_object_resizing(
                object_size_info=obj_size_info,
                goal_task=task_obj_output_info['task'],
            )

            # Query GPT
            gpt_text_response = gpt(task_object_resizing_payload, verbose=self.verbose)

            if gpt_text_response is None:
                # Failed, terminate early
                return False, None

            # Parse GPT result
            gpt_task_obj_resize_result = self.parse_string(input_str=gpt_text_response)
            # gpt_task_obj_resize_result = {"food": 0.3}

            task_obj_resize_info = copy.deepcopy(task_obj_output_info)
            for obj_name, obj_info in task_obj_resize_info['objects'].items():
                # calculate scale factor
                scale_factor = gpt_task_obj_resize_result[obj_name] / max(obj_size_info[obj_name])

                obj_info['scale'] = [1 * scale_factor for _ in range(3)]
                # obj_info['scale_factor'] = scale_factor

            print(f"task_obj_resize_info: {task_obj_resize_info}")
            scenario_obj_num = os.path.basename(scenario_obj_num_json_path).replace("step_5_output_info_", "").replace(".json", "")
            task_object_resizing_path = f"{save_dir}/target_object_resizing_output_info_{scenario_obj_num}.json"
            json_list.append(task_object_resizing_path)

            with open(task_object_resizing_path, "w+") as f:
                # json.dump(task_obj_resize_info, f, indent=4)
                write_json_like(task_obj_resize_info, f, indent=0)
            


        with open(f"{save_dir}/task_obj_output_info.json", "w+") as f:
        # json.dump(task_extraction_output_info, f, indent=4, 
        #           cls=OneLineListEncoder)
            json.dump(json_list, f, indent=4)
        print("""

##########################################
### Completed Task Object Resizing! ###
##########################################

        """)

        og.shutdown()
        return True, task_object_resizing_path

    def parse_string(self, input_str):
        # 입력 문자열을 줄 단위로 분리
        lines = input_str.strip().split('\n')
        
        # 첫 번째 줄(설명)을 제거하고 나머지 줄로 작업
        content_lines = lines[1:]
        
        # 결과 딕셔너리 초기화
        result = {}
        
        # 각 줄을 파싱하여 딕셔너리에 추가
        for line in content_lines:
            # 빈 줄 건너뛰기
            if not line.strip():
                continue
            
            # 괄호 안의 숫자와 객체 이름 추출
            parts = line.split('(')
            if len(parts) < 2:
                continue
            
            obj_name = parts[0].strip()
            # 괄호를 제거하고 숫자만 추출
            number_str = parts[1].split(')')[0].strip()
            
            # 문자열을 적절한 숫자 타입으로 변환 (정수 또는 부동소수점)
            try:
                number = float(number_str)

                # 결과 딕셔너리에 추가
                result[obj_name] = number
            except ValueError:
                # 숫자 변환에 실패한 경우 건너뛰기
                continue
        
        return result
    
def write_json_like(data, file, indent=0):
    spacing = " " * indent

    if isinstance(data, dict):
        file.write("{\n")
        items = list(data.items())
        for i, (key, value) in enumerate(items):
            file.write(" " * (indent + 4) + f'"{key}": ')
            write_json_like(value, file, indent + 4)
            if i < len(items) - 1:
                file.write(",")
            file.write("\n")
        file.write(spacing + "}")

    elif isinstance(data, list):
        if all(isinstance(i, (int, float, str, bool, type(None))) for i in data):
            # 1D list → 한 줄
            file.write("[" + ",".join(json_safe_repr(i) for i in data) + "]")
        else:
            # 2D or 3D list → 줄바꿈 + 들여쓰기
            file.write("[\n")
            for i, item in enumerate(data):
                file.write(" " * (indent + 4))
                write_json_like(item, file, indent + 4)
                if i < len(data) - 1:
                    file.write(",")
                file.write("\n")
            file.write(spacing + "]")

    elif isinstance(data, str):
        file.write(f'"{data}"')

    elif data is None:
        file.write("null")

    elif isinstance(data, bool):
        file.write("true" if data else "false")

    else:
        file.write(str(data))

def json_safe_repr(x):
    if isinstance(x, str):
        return '"' + x.replace('"', '\\"') + '"'
    elif isinstance(x, (int, float)):
        return str(x)
    elif isinstance(x, list):
        return "[" + ",".join(json_safe_repr(i) for i in x) + "]"
    elif isinstance(x, dict):
        return "{" + ",".join(f'"{k}":{json_safe_repr(v)}' for k, v in x.items()) + "}"
    else:
        return repr(x)
