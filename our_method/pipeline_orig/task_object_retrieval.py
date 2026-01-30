import enum
from hmac import new
from math import dist
from sklearn import feature_extraction
import torch
from torchvision.ops.boxes import _box_xyxy_to_cxcywh
from groundingdino.util.inference import load_image
import numpy as np
from pathlib import Path
from PIL import Image
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
RE_AXIS_MAT = [
    [[0, -1, 0], [1, 0, 0], [0, 0, 1]],   # 0도
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],    # 90도
    [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],   # 180도
    [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]   # 270도
]


class TaskObjectRetrieval:
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
            feature_matcher,
            verbose=False,
    ):
        """
        Args:
            feature_matcher (FeatureMatcher): Feature matcher class instance to use for segmenting objects
                and matching features
            verbose (bool): Whether to display verbose print outs during execution or not
        """
        self.fm = feature_matcher
        self.fm.eval()
        self.verbose = verbose
        self.device = self.fm.device

    def __call__(
            self,
            step_1_output_path,
            step_2_output_path,
            step_3_output_path,
            task_spatial_reasoning_output_path,
            gpt_api_key,
            gpt_version="4o",
            top_k_categories=3,
            top_k_models=5,
            n_digital_cousins=3,
            n_cousins_reselect_cand=3,
            remove_background=False,
            gpt_select_cousins=True,
            n_cousins_link_count_threshold=3,
            save_dir=None,
            start_at_name=None,
            find_front_view = True,
            use_distractor_noise=False,
            use_distractor_category=5,
            distractor_top_k=3,
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
        assert n_digital_cousins <= top_k_models, \
            f"n_digital_cousins ({n_digital_cousins}) cannot be greater than top_k_models ({top_k_models})!"

        # Parse save_dir, and create the directory if it doesn't exist
        if save_dir is None:
            save_dir = os.path.dirname(os.path.dirname(task_spatial_reasoning_output_path))
        save_dir = os.path.join(save_dir, "task_object_retrieval")
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(f"Computing digital cousins given output {task_spatial_reasoning_output_path}...")

        if self.verbose:
            print("""

##################################################################
### 1. Use CLIP embeddings to find top-K categories per object ###
##################################################################

            """)

        # Load meta info
        with open(task_spatial_reasoning_output_path, "r") as f:
            task_spatial_reasoning_output_info = json.load(f)              

        with open(step_3_output_path, "r") as f:
            step_3_output_info = json.load(f)

        json_list = []
        # print(f"task_spatial_reasoning_output_info : {task_spatial_reasoning_output_info}")
        # print(f"task_spatial_reasoning_output_info : {type(task_spatial_reasoning_output_info)}")
        for scenario_json_path in task_spatial_reasoning_output_info:
            with open(scenario_json_path, "r") as f:
                task_extraction_output_info = json.load(f)
            
            # 'objects' 안에 있는 키 모음
            existing_objects = set(task_extraction_output_info['objects'].keys())

            # 각 object에 대해 'new'를 추가
            for obj_name, obj_info in task_extraction_output_info['objects'].items():
                parent_obj = obj_info.get('parent_object')
                obj_info['new'] = parent_obj in existing_objects
            
            # 정렬: new=False 먼저, new=True 나중
            sorted_objects = dict(
                sorted(
                    task_extraction_output_info['objects'].items(),
                    key=lambda item: item[1]['new']
                )
            )

            # 다시 넣기
            task_extraction_output_info['objects'] = sorted_objects

            ##################################
            obj_name_list = []
            for obj_name, obj_info in task_extraction_output_info["objects"].items():
                obj_name_list.append(obj_name)
                # print(obj_name)
                # print(obj_info)

            all_categories = list(get_all_dataset_categories(do_not_include_categories=DO_NOT_MATCH_CATEGORIES, replace_underscores=True))
            

            clip = CLIPEncoder(backbone_name="ViT-B/32", device=self.device)
            res = faiss.StandardGpuResources()
            index_flat = faiss.IndexFlatL2(clip.embedding_dim)
            gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
            selected_categories = dict()

            # TODO
            obj_phrases = [i for i in obj_name_list]

            if len(obj_phrases) > 0:
                if self.verbose:
                    print(f"Computing top-{top_k_categories} for phrases using CLIP...")
                
                # CLIP 임베딩 계산
                text_features = clip.get_text_features(text=all_categories)
                cand_text_features = clip.get_text_features(text=obj_name_list) # 1800개

                # FAISS를 이용한 최근접 이웃 탐색
                gpu_index_flat.reset()
                gpu_index_flat.add(text_features)
                dists, idxs = gpu_index_flat.search(cand_text_features, top_k_categories)

                # 결과 매핑
                for obj_idx, topk_idxs in zip(range(len(obj_phrases)), idxs):
                    top_k_cat_names = [all_categories[topk_idx] for topk_idx in topk_idxs]
                    selected_categories[obj_name_list[obj_idx]] = top_k_cat_names
            
            # Store these results
            topk_categories_info = {
                "topk_categories": selected_categories,
            }
            
            topk_categories_path = f"{save_dir}/task_topk_categories.json"
            
            with open(topk_categories_path, "w+") as f:
                json.dump(topk_categories_info, f, indent=4)

            # Clean up resources to avoid OOM error
            del res
            del clip
            del gpu_index_flat
            ##################################
                # exit()

                    

            if self.verbose:
                print("""

##############################################################
### 2. Select Task Object using GPT ###
##############################################################

                """)
            # Create GPT instance
            # assert gpt_api_key is not None, "gpt_api_key must be specified in order to use GPT model!"
            # gpt = GPT(api_key=gpt_api_key, version=gpt_version)
            # input_sim_rgb_path = step_3_output_info["scene_0"]["scene_img"]
            # input_real_rgb_path = step_3_output_info["scene_0"]["scene_img"]
            input_sim_real_rgb_path = os.path.join(os.path.dirname(step_3_output_info["scene_0"]["scene_graph"]), "scene_0_visualization.png")

            should_start = start_at_name is None
            n_instances = len(obj_name_list)

            # Create GPT instance
            assert gpt_api_key is not None, "gpt_api_key must be specified in order to use GPT model!"
            gpt = GPT(api_key=gpt_api_key, version=gpt_version, log_dir_tail="_TaskObjectRetrieval")

            
            for instance_idx, name in enumerate(obj_name_list):
                # Skip if starting at name has not been reached yet
                if not should_start:
                    if start_at_name == name:
                        should_start = True
                    else:
                        # Immediately keep looping
                        continue
                
                og_categories = selected_categories[name]
                obj_save_dir = f"{save_dir}/{name}"
                topk_model_candidates_dir = f"{obj_save_dir}/top_k_model_candidates"
                Path(topk_model_candidates_dir).mkdir(parents=True, exist_ok=True)
                
                category_list = []
                model_list =[]
                
                selected_models = set()


                # Find Top-K candidates
                candidate_imgs_fdirs = [f"{digital_cousins.ASSET_DIR}/objects/{og_category.replace(' ', '_')}/snapshot" for og_category in og_categories]
                
                candidate_imgs = list(sorted(f"{candidate_imgs_fdir}/{model}"
                                for candidate_imgs_fdir in candidate_imgs_fdirs
                                for model in os.listdir(candidate_imgs_fdir)
                                if model not in selected_models))

                concat_img_save_dir = os.path.join(topk_model_candidates_dir, f'{name}_candidate_input_visualization.png')
                concat_img = self.make_concat_images(
                    snapshot_imgs_path=candidate_imgs,
                    visualize_resolution=(1280, 960),  # 해상도 조절 가능
                    images_per_row=4,
                    save_path=concat_img_save_dir
                )
                
                if task_extraction_output_info["objects"][name]["new"]:
                    nn_selection_payload = gpt.payload_nearest_neighbor_text_ref_scene(
                                        sim_real_img_path=input_sim_real_rgb_path,
                                        # parent_obj_bbox_img_path=f"{os.path.dirname(step_1_output_path)}/segmented_objects/{task_extraction_output_info['objects'][name]['parent_object']}_annotated_bboxes.png",
                                        goal_task=task_extraction_output_info["task"],
                                        parent_obj_name=task_extraction_output_info["objects"][name]["parent_object"],
                                        placement=task_extraction_output_info["objects"][name]["placement"],
                                        caption=obj_phrases[instance_idx],
                                        candidates_path=concat_img_save_dir,
                                        top_k=top_k_models)
                else:               
                    nn_selection_payload = gpt.payload_nearest_neighbor_text_ref_scene_bbox(
                                        sim_real_img_path=input_sim_real_rgb_path,
                                        parent_obj_bbox_img_path=f"{os.path.dirname(step_1_output_path)}/segmented_objects/{task_extraction_output_info['objects'][name]['parent_object']}_annotated_bboxes.png",
                                        goal_task=task_extraction_output_info["task"],
                                        parent_obj_name=task_extraction_output_info["objects"][name]["parent_object"],
                                        placement=task_extraction_output_info["objects"][name]["placement"],
                                        caption=obj_phrases[instance_idx],
                                        candidates_path=concat_img_save_dir,
                                        top_k=top_k_models)
                
                
                gpt_text_response = gpt(nn_selection_payload)

                print("GPT Response :")
                print(f"   {gpt_text_response}")

                if gpt_text_response is None:
                    print(f"gpt_text_response is None")
                    # Failed, terminate early
                    return False, None
                # 숫자 모두 추출
                matches = re.findall(r'\b\d+\b', gpt_text_response)

                print("Extract number list :")
                print(f"   {matches}")

                # 최대 top_k개만 선택
                nn_model_indices = [int(m) for m in matches[:top_k_models]]  # 0-based 인덱스로 변환
                print("final number list :")
                print(f"   {nn_model_indices}\n")


                # # # 숫자가 하나도 없을 경우 → 실패 처리
                # # if not matches:
                # #     return False, None
                # if name == "cup":
                #     nn_model_indices = [18, 6, 27]
                # else : 
                #     nn_model_indices = [1, 2, 3]
                
                # 후보 이미지 리스트에서 선택된 인덱스만 추출
                n_candidates = [candidate_imgs[i-1] for i in nn_model_indices]
                
                results = {
                    "k": top_k_models,
                    "candidates": n_candidates,
                }

                with open(f"{topk_model_candidates_dir}/{name}_feature_matcher_results.json", "w+") as f:
                    json.dump(results, f)

                concat_img_save_dir = os.path.join(topk_model_candidates_dir, f'{name}_candidate_gpt_results_visualization.png')
                concat_img = self.make_concat_images(
                    snapshot_imgs_path=n_candidates,
                    visualize_resolution=(1280, 960),  # 해상도 조절 가능
                    images_per_row=3,
                    save_path=concat_img_save_dir,
                    fontscale = 0
                )

                for path in n_candidates:
                    category = path.split("/")[-3]  # snapshot 바로 앞 폴더
                    model = path.split("/")[-1].split(".")[0].split("_")[-1]  # 파일 이름에서 모델명만
                    category_list.append(category)
                    model_list.append(model)

                task_extraction_output_info["objects"][name]["category"] = category_list
                task_extraction_output_info["objects"][name]["model"] = model_list


                # 정면 포즈 찾기

            print(f"task_extraction_output_info: {task_extraction_output_info}")

            if self.verbose:
                print("""

##############################################################
### 3. Select Task Object Front Position using GPT ###
##############################################################

                """)

            for instance_idx, name in enumerate(obj_name_list):
                category_list = task_extraction_output_info["objects"][name]["category"]
                model_list = task_extraction_output_info["objects"][name]["model"]
                re_axis_mat_list = []
                obj_save_dir = f"{save_dir}/{name}"
                front_pose_select_dir = f"{obj_save_dir}/task_object_front_pose_select"
                Path(front_pose_select_dir).mkdir(parents=True, exist_ok=True)

                results = {}

                for model_idx, model_name in enumerate(model_list):
                    print(f"category: {category_list[model_idx]}")
                    print(f"model: {model_list[model_idx]}")
                    
                    # Find Top-K candidates
                    candidate_model_view_fdirs = f"{digital_cousins.ASSET_DIR}/objects/{category_list[model_idx]}/model/{model_list[model_idx]}" 

                    candidate_model_view_imgs = sorted(
                        os.path.join(candidate_model_view_fdirs, fname)
                        for fname in os.listdir(candidate_model_view_fdirs)
                        if fname.endswith('.png') and int(fname.rstrip('.png').split('_')[-1]) % 25 == 0
                    )

                    if find_front_view:
                        concat_img_save_dir = os.path.join(front_pose_select_dir, f'{name}_{model_name}_candidate_view_input_visualization.png')

                        concat_img = self.make_concat_images(
                            snapshot_imgs_path=candidate_model_view_imgs,
                            visualize_resolution=(640, 480),  # 해상도 조절 가능
                            images_per_row=4,
                            save_path=concat_img_save_dir
                        ) 

                        nn_selection_payload = gpt.payload_front_view_image(
                                candidate_view_path=concat_img_save_dir,
                                goal_task=task_extraction_output_info["task"],
                                parent_obj_name=task_extraction_output_info["objects"][name]["parent_object"],
                                placement=task_extraction_output_info["objects"][name]["placement"],
                                caption=obj_phrases[instance_idx]
                                )
                        
                        gpt_text_response = gpt(nn_selection_payload)
                        print(f"gpt_text_response: {gpt_text_response}")
                        if gpt_text_response is None:
                            print(f"gpt_text_response is None")
                            # Failed, terminate early
                            return False, None

                        # Extract the first non-negative integer from the response
                        match = re.search(r'\b\d+\b', gpt_text_response)

                        if match:
                            nn_model_index = int(match.group()) - 1
                        else:
                            print(f"match is empty")
                            # # No valid integer found, handle this case
                            # return False, None
                            nn_model_index = 0
                    else:
                        nn_model_index = 1
                    
                    # nn_model_index = 1
                    
                    results[model_name] = {
                        "view_path": candidate_model_view_imgs[nn_model_index],
                        "re_axis_mat": RE_AXIS_MAT[nn_model_index],
                    }
                    
                    re_axis_mat_list.append(RE_AXIS_MAT[nn_model_index])
                    
                    shutil.copy(candidate_model_view_imgs[nn_model_index], 
                                os.path.join(front_pose_select_dir, os.path.basename(candidate_model_view_imgs[nn_model_index])))
                    
                with open(f"{front_pose_select_dir}/model_pose_selection_results.json", "w+") as f:
                    json.dump(results, f)
                
                task_extraction_output_info["objects"][name]["re_axis_mat"] = re_axis_mat_list

            print(f"task_extraction_output_info: {task_extraction_output_info}")

            if self.verbose:
                print("""

##############################################################
### 4. Select Parent Object Front Position using GPT ###
##############################################################

                """)
            for instance_idx, name in enumerate(obj_name_list):
                if task_extraction_output_info["objects"][name]["new"]:
                    continue
                # category_list = task_extraction_output_info["objects"][name]["category"]
                # model_list = task_extraction_output_info["objects"][name]["model"]
                # re_axis_mat_list = []
                # print(task_extraction_output_info)
                parent_object_name = task_extraction_output_info["objects"][name]["parent_object"]
                parent_object_category = step_3_output_info["scene_0"]["objects"][parent_object_name]["category"]
                parent_object_model = step_3_output_info["scene_0"]["objects"][parent_object_name]["model"]
                obj_save_dir = f"{save_dir}/{name}"
                parent_front_pose_select_dir = f"{obj_save_dir}/parent_object_front_pose_select"
                Path(parent_front_pose_select_dir).mkdir(parents=True, exist_ok=True)

                results = {}
                parent_candidate_model_view_fdirs = f"{digital_cousins.ASSET_DIR}/objects/{parent_object_category}/model/{parent_object_model}" 
                parent_candidate_model_view_imgs = sorted(
                        os.path.join(parent_candidate_model_view_fdirs, fname)
                        for fname in os.listdir(parent_candidate_model_view_fdirs)
                        if fname.endswith('.png') and int(fname.rstrip('.png').split('_')[-1]) % 25 == 0
                    )
                
                if find_front_view:
                    concat_img_save_dir = os.path.join(parent_front_pose_select_dir, f'{name}_parent_object_{model_name}_candidate_view_input_visualization.png')
                    concat_img = self.make_concat_images(
                            snapshot_imgs_path=parent_candidate_model_view_imgs,
                            visualize_resolution=(640, 480),  # 해상도 조절 가능
                            images_per_row=4,
                            save_path=concat_img_save_dir
                        )
                    
                    nn_selection_payload = gpt.payload_front_view_image(
                                candidate_view_path=concat_img_save_dir,
                                goal_task=task_extraction_output_info["task"],
                                parent_obj_name=task_extraction_output_info["objects"][name]["parent_object"],
                                placement=task_extraction_output_info["objects"][name]["placement"],
                                caption=obj_phrases[instance_idx]
                                )
                        
                    gpt_text_response = gpt(nn_selection_payload)
                    if gpt_text_response is None:
                        # Failed, terminate early
                        return False, None

                    # Extract the first non-negative integer from the response
                    match = re.search(r'\b\d+\b', gpt_text_response)

                    if match:
                        nn_model_index = int(match.group()) - 1
                    else:
                        # No valid integer found, handle this case
                        return False, None
                else:
                    nn_model_index = 1
                
                # nn_model_index = 1
                
                results[model_name] = {
                    "view_path": parent_candidate_model_view_imgs[nn_model_index],
                    "re_axis_mat": RE_AXIS_MAT[nn_model_index],
                }
                
                shutil.copy(parent_candidate_model_view_imgs[nn_model_index], 
                            os.path.join(parent_front_pose_select_dir, os.path.basename(parent_candidate_model_view_imgs[nn_model_index])))
                
                with open(f"{parent_front_pose_select_dir}/parent_model_pose_selection_results.json", "w+") as f:
                    json.dump(results, f)
                
                task_extraction_output_info["objects"][name]["parent_re_axis_mat"] = RE_AXIS_MAT[nn_model_index]

            step_5_output_path = f"{save_dir}/step_5_output_info.json"

  

            
            
            index_lists = [
                range(len(task_extraction_output_info["objects"][obj]["model"]))
                for obj in obj_name_list
            ]

            # 모든 조합 만들기
            for num, idxs in enumerate(product(*index_lists)):
                combo_obj_data = {}

                for i, obj_name in enumerate(obj_name_list):
                    idx = idxs[i]
                    obj = task_extraction_output_info["objects"][obj_name]

                    # 전체 복사해서 변경하는 방식 (원본 유지)
                    new_obj = copy.deepcopy(obj)

                    # 동적으로 바꾸는 항목만 교체
                    new_obj["model"] = obj["model"][idx]
                    new_obj["category"] = obj["category"][idx]
                    new_obj["re_axis_mat"] = obj["re_axis_mat"][idx]
                    if new_obj["new"]:
                        new_obj["parent_re_axis_mat"] = combo_obj_data[obj["parent_object"]]["re_axis_mat"]
                    new_obj.pop("new", None)

                    combo_obj_data[obj_name] = new_obj


                output_json = {
                    "task": task_extraction_output_info["task"],
                    "objects": combo_obj_data
                }

                scenario_num = os.path.basename(scenario_json_path).replace("task_obj_output_info_", "").replace(".json", "")
                json_path = f"{save_dir}/step_5_output_info_{scenario_num}_{num}.json"
                json_list.append(json_path)
                with open(json_path, "w+") as f:
                # json.dump(task_extraction_output_info, f, indent=4, 
                #           cls=OneLineListEncoder)
                    write_json_like(output_json, f, indent=0)

            # TODO
            if use_distractor_noise:

                print("""

##########################################
### Start Distractor Object Matching! ###
##########################################

                """)
                input_sim_real_rgb_path = os.path.join(os.path.dirname(step_3_output_info["scene_0"]["scene_graph"]), "scene_0_visualization.png")


                # Create GPT instance
                # assert gpt_api_key is not None, "gpt_api_key must be specified in order to use GPT model!"
                # gpt = GPT(api_key=gpt_api_key, version=gpt_version, log_dir_tail="_TaskObjectRetrieval")
                # Get the object phrases from the task extraction output info

                parent_objects_inside = [
                    obj_info['parent_object']
                    for obj_info in task_extraction_output_info['objects'].values()
                    if obj_info.get('placement') == 'inside'
                ]
                distractor_topk_categories_info = {}
                for parent_object_category in parent_objects_inside:
                    nn_selection_payload = gpt.payload_distractor_inside_object_category(
                                                sim_real_img_path=input_sim_real_rgb_path,
                                                # parent_obj_bbox_img_path=f"{os.path.dirname(step_1_output_path)}/segmented_objects/{task_extraction_output_info['objects'][name]['parent_object']}_annotated_bboxes.png",
                                                goal_task=task_extraction_output_info["task"],
                                                parent_obj_name=parent_object_category,
                                                use_distractor_category=use_distractor_category)
                    
                    gpt_text_response = gpt(nn_selection_payload)

                    print("GPT Response :")
                    print(f"   {gpt_text_response}")
                    # GPT Response :
                    # pencil case, pens, notebooks, folders, stapler


                    if gpt_text_response is None:
                        print(f"gpt_text_response is None")
                        # Failed, terminate early
                        return False, None

                    # 문자열을 쉼표(,) 기준으로 나눈 뒤, strip으로 양쪽 공백 제거
                    category_list = [cat.strip() for cat in gpt_text_response.split(",") if cat.strip() != ""]

                    # ============================================
                    # 2. CLIP + FAISS 를 이용해 유사 category 찾기
                    # ============================================

                    # 전체 데이터셋에서 사용할 category 불러오기
                    all_categories = list(get_all_dataset_categories(
                        do_not_include_categories=DO_NOT_MATCH_CATEGORIES,
                        replace_underscores=True
                    ))

                    # CLIP & FAISS 초기화
                    clip = CLIPEncoder(backbone_name="ViT-B/32", device=self.device)
                    res = faiss.StandardGpuResources()
                    index_flat = faiss.IndexFlatL2(clip.embedding_dim)
                    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)

                    if self.verbose:
                        print(f"🔍 Finding top-{top_k_categories} similar categories for GPT outputs using CLIP+FAISS...")
                        print(f"   GPT Categories: {category_list}")

                    # 텍스트 임베딩 계산
                    all_cat_features = clip.get_text_features(text=all_categories)
                    query_features = clip.get_text_features(text=category_list)

                    # FAISS 검색
                    gpu_index_flat.reset()
                    gpu_index_flat.add(all_cat_features)
                    dists, idxs = gpu_index_flat.search(query_features, top_k_categories)

                    # 결과 매핑
                    selected_distractor_categories = {}
                    for i, topk_idxs in enumerate(idxs):
                        topk_names = [all_categories[j] for j in topk_idxs]
                        selected_distractor_categories[category_list[i]] = topk_names

                    # 저장
                    distractor_topk_categories_info = {
                        f"{parent_object_category}": selected_distractor_categories
                    }

                    topk_categories_path = os.path.join(save_dir, "gpt_distractor_topk_categories.json")
                    with open(topk_categories_path, "w") as f:
                        json.dump(distractor_topk_categories_info, f, indent=4)


                    # 자원 정리
                    del clip
                    del res
                    del gpu_index_flat
                
                distractor_extraction_output_info = {}
                for parent_object_name in distractor_topk_categories_info:
                    distractor_categories = distractor_topk_categories_info[parent_object_name]
                    parent_obj_save_dir = f"{save_dir}/distactor/{parent_object_name}"

                    # 이미 폴더가 있으면 건너뛰기
                    if os.path.exists(parent_obj_save_dir):
                        continue
                    obj_name_list = []
                    for distractor_object_name in distractor_categories.keys():
                        og_categories = distractor_categories[distractor_object_name]
                        distractor_parent_obj_save_dir = f"{parent_obj_save_dir}/{distractor_object_name}"
                        topk_model_candidates_dir = f"{distractor_parent_obj_save_dir}/top_k_model_candidates"
                        Path(topk_model_candidates_dir).mkdir(parents=True, exist_ok=True)

                        obj_name_list.append(distractor_object_name)

                        category_list = []
                        model_list =[]
                        
                        selected_models = set()


                        # Find Top-K candidates
                        candidate_imgs_fdirs = [f"{digital_cousins.ASSET_DIR}/objects/{og_category.replace(' ', '_')}/snapshot" for og_category in og_categories]
                        
                        candidate_imgs = list(sorted(f"{candidate_imgs_fdir}/{model}"
                                        for candidate_imgs_fdir in candidate_imgs_fdirs
                                        for model in os.listdir(candidate_imgs_fdir)
                                        if model not in selected_models))

                        concat_img_save_dir = os.path.join(topk_model_candidates_dir, f'{distractor_object_name}_candidate_input_visualization.png')
                        concat_img = self.make_concat_images(
                            snapshot_imgs_path=candidate_imgs,
                            visualize_resolution=(1280, 960),  # 해상도 조절 가능
                            images_per_row=5,
                            save_path=concat_img_save_dir
                        )
                        nn_selection_payload = gpt.payload_nearest_neighbor_text_ref_scene_bbox(
                                            sim_real_img_path=input_sim_real_rgb_path,
                                            parent_obj_bbox_img_path=f"{os.path.dirname(step_1_output_path)}/segmented_objects/{parent_object_name}_annotated_bboxes.png",
                                            goal_task=task_extraction_output_info["task"],
                                            parent_obj_name=parent_object_name,
                                            placement="inside",
                                            caption=distractor_object_name,
                                            candidates_path=concat_img_save_dir,
                                            top_k=distractor_top_k)
                        
                        gpt_text_response = gpt(nn_selection_payload)

                        print("GPT Response :")
                        print(f"   {gpt_text_response}")

                        if gpt_text_response is None:
                            print(f"gpt_text_response is None")
                            # Failed, terminate early
                            return False, None
                        # 숫자 모두 추출
                        matches = re.findall(r'\b\d+\b', gpt_text_response)

                        print("Extract number list :")
                        print(f"   {matches}")

                        # 최대 top_k개만 선택
                        nn_model_indices = [int(m) for m in matches[:distractor_top_k]]  # 0-based 인덱스로 변환
                        print("final number list :")
                        print(f"   {nn_model_indices}\n")


                        # # # 숫자가 하나도 없을 경우 → 실패 처리
                        # # if not matches:
                        # #     return False, None
                        # if name == "cup":
                        #     nn_model_indices = [18, 6, 27]
                        # else : 
                        #     nn_model_indices = [1, 2, 3]
                        
                        # 후보 이미지 리스트에서 선택된 인덱스만 추출
                        n_candidates = [candidate_imgs[i-1] for i in nn_model_indices]
                        
                        results = {
                            "k": distractor_top_k,
                            "candidates": n_candidates,
                        }

                        with open(f"{topk_model_candidates_dir}/{distractor_object_name}_feature_matcher_results.json", "w+") as f:
                            json.dump(results, f)

                        for path in n_candidates:
                            category = path.split("/")[-3]  # snapshot 바로 앞 폴더
                            model = path.split("/")[-1].split(".")[0].split("_")[-1]  # 파일 이름에서 모델명만
                            category_list.append(category)
                            model_list.append(model)
                        
                        if parent_object_name not in distractor_extraction_output_info:
                            distractor_extraction_output_info[parent_object_name] = {}

                        if distractor_object_name not in distractor_extraction_output_info[parent_object_name]:
                            distractor_extraction_output_info[parent_object_name][distractor_object_name] = {}


                        distractor_extraction_output_info[parent_object_name][distractor_object_name]["category"] = category_list
                        distractor_extraction_output_info[parent_object_name][distractor_object_name]["model"] = model_list


                            
                        re_axis_mat_list = []
                        front_pose_select_dir = f"{distractor_parent_obj_save_dir}/task_object_front_pose_select"
                        Path(front_pose_select_dir).mkdir(parents=True, exist_ok=True)

                        #     results = {}

                        for model_idx, model_name in enumerate(model_list):
                            print(f"category: {category_list[model_idx]}")
                            print(f"model: {model_list[model_idx]}")
                                
                            # # Find Top-K candidates
                            candidate_model_view_fdirs = f"{digital_cousins.ASSET_DIR}/objects/{category_list[model_idx]}/model/{model_list[model_idx]}" 

                            candidate_model_view_imgs = sorted(
                                os.path.join(candidate_model_view_fdirs, fname)
                                for fname in os.listdir(candidate_model_view_fdirs)
                                if fname.endswith('.png') and int(fname.rstrip('.png').split('_')[-1]) % 25 == 0
                            )

                            concat_img_save_dir = os.path.join(front_pose_select_dir, f'{distractor_object_name}_{model_name}_candidate_view_input_visualization.png')

                            concat_img = self.make_concat_images(
                                snapshot_imgs_path=candidate_model_view_imgs,
                                visualize_resolution=(640, 480),  # 해상도 조절 가능
                                images_per_row=4,
                                save_path=concat_img_save_dir
                            ) 

                            nn_selection_payload = gpt.payload_front_view_image(
                                    candidate_view_path=concat_img_save_dir,
                                    goal_task=task_extraction_output_info["task"],
                                    parent_obj_name=parent_object_name,
                                    placement="inside",
                                    caption=distractor_object_name
                                    )
                            
                            gpt_text_response = gpt(nn_selection_payload)
                            print(f"gpt_text_response: {gpt_text_response}")
                            if gpt_text_response is None:
                                print(f"gpt_text_response is None")
                                # Failed, terminate early
                                return False, None

                            # Extract the first non-negative integer from the response
                            match = re.search(r'\b\d+\b', gpt_text_response)

                            if match:
                                nn_model_index = int(match.group()) - 1
                            else:
                                print(f"match is empty")
                                # # No valid integer found, handle this case
                                # return False, None
                                nn_model_index = 0

                            
                            # nn_model_index = 1
                            
                            results[model_name] = {
                                "view_path": candidate_model_view_imgs[nn_model_index],
                                "re_axis_mat": RE_AXIS_MAT[nn_model_index],
                            }
                            
                            re_axis_mat_list.append(RE_AXIS_MAT[nn_model_index])
                            print(re_axis_mat_list)
                            
                            shutil.copy(candidate_model_view_imgs[nn_model_index], 
                                        os.path.join(front_pose_select_dir, os.path.basename(candidate_model_view_imgs[nn_model_index])))
                            
                        with open(f"{front_pose_select_dir}/model_pose_selection_results.json", "w+") as f:
                            json.dump(results, f)
                        
                        distractor_extraction_output_info[parent_object_name][distractor_object_name]["re_axis_mat"] = re_axis_mat_list

                    print(distractor_extraction_output_info)       

                    index_lists = [
                        range(len(distractor_extraction_output_info[parent_object_name][obj]['model']))
                        for obj in obj_name_list
                    ]    
                    distractor_json_list = []

                    # 모든 조합 생성
                    for num, idxs in enumerate(product(*index_lists)):
                        combo_obj_data = {}

                        for i, obj_name in enumerate(obj_name_list):
                            idx = idxs[i]
                            obj = distractor_extraction_output_info[parent_object_name][obj_name]

                            # 원본 복사
                            new_obj = copy.deepcopy(obj)

                            # 현재 idx에 맞는 값으로 교체
                            new_obj["model"] = obj["model"][idx]
                            new_obj["category"] = obj["category"][idx]
                            new_obj["re_axis_mat"] = obj["re_axis_mat"][idx]


                            obj_name = obj_name.replace(" ", "_")
                            combo_obj_data[obj_name] = new_obj

                        # 저장할 JSON 구조
                        output_json = {
                            "task": task_extraction_output_info["task"],  # task 그대로 유지
                            "objects": combo_obj_data
                        }

                        # 파일명 규칙 유지
                        json_path = f"{parent_obj_save_dir}/distractor_output_info_{num}.json"

                        # 저장
                        distractor_json_list.append(json_path)
                        Path(save_dir).mkdir(parents=True, exist_ok=True)
                        with open(json_path, "w+") as f:
                            write_json_like(output_json, f, indent=0)

                    with open(f"{parent_obj_save_dir}/distractor_output_info.json", "w+") as f:
                    # json.dump(task_extraction_output_info, f, indent=4, 
                    #           cls=OneLineListEncoder)
                        json.dump(distractor_json_list, f, indent=4)
        print("""

##########################################
### Completed Task Object Matching! ###
##########################################

        """)
        
        with open(f"{save_dir}/task_obj_output_info.json", "w+") as f:
        # json.dump(task_extraction_output_info, f, indent=4, 
        #           cls=OneLineListEncoder)
            json.dump(json_list, f, indent=4)

        

        return True, step_2_output_path



    def make_concat_images(self, snapshot_imgs_path, visualize_resolution=(640, 480), images_per_row=10, fontscale = 4, save_path=None):
        """
        snapshot_list_files 내 이미지들을 한 줄에 10개씩 정렬하고, 왼쪽 위에 파일명 숫자 라벨을 추가.

        Args:
            snapshot_path (str): 이미지가 들어있는 디렉토리 경로
            snapshot_list_files (list): 이미지 파일 이름 리스트
            visualize_resolution (tuple): 각 이미지 리사이즈 크기
            images_per_row (int): 한 줄에 넣을 이미지 수
            save_path (str or None): 저장 경로. None이면 저장하지 않음

        Returns:
            np.array: 합쳐진 이미지 배열 (H_total, W_total, 3)
        """
        resized_labeled_imgs = []

        for i, img_path in enumerate(snapshot_imgs_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img is None:
                print(f"[경고] 이미지를 불러올 수 없습니다: {img_path}")
                continue
            img_resized = cv2.resize(img, visualize_resolution)

            # 텍스트 추가
            cv2.putText(
                img_resized,
                str(i+1),
                org=(5, int(30 * fontscale)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=fontscale,
                color=(255, 255, 255),
                # color=(0, 0, 0),
                thickness=2,
                lineType=cv2.LINE_AA
            )

            # 🔍 디버깅: 리사이즈된 이미지의 shape 출력
            resized_labeled_imgs.append(img_resized)

        # Grid로 연결
        rows = []
        for i in range(0, len(resized_labeled_imgs), images_per_row):
            row_imgs = resized_labeled_imgs[i:i + images_per_row]
                    # 마지막 줄 padding
            if len(row_imgs) < images_per_row:
                padding = images_per_row - len(row_imgs)
                blank_img = np.zeros((visualize_resolution[1], visualize_resolution[0], 3), dtype=np.uint8)
                row_imgs += [blank_img] * padding

            row = np.concatenate(row_imgs, axis=1)
            rows.append(row)
        # full_img = np.concatenate(rows, axis=0)

        try:
            full_img = np.concatenate(rows, axis=0)
        except Exception as e:
            print(f"[에러] 전체 이미지 연결 중 오류 발생: {e}")
            return None

        if save_path:
            Image.fromarray(full_img).save(save_path)

        return full_img

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
