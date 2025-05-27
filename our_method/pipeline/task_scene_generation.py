from calendar import c
from pickle import FALSE
from tabnanny import check
from threading import local
from turtle import distance

from robomimic.utils.log_utils import PrintLogger
import torch as th
import numpy as np
from pathlib import Path
from PIL import Image
from copy import deepcopy
import os
import re
import json
import imageio
from typing import Optional, List
import omnigibson as og
from our_method.models.gpt import GPT
from omnigibson.scenes import Scene
from omnigibson.objects import DatasetObject
from omnigibson.object_states import Touching
from omnigibson.object_states import ToggledOn
from omnigibson.utils.sim_utils import place_base_pose, check_collision
import random
import our_method
from our_method.utils.processing_utils import NumpyTorchEncoder, unprocess_depth_linear, compute_point_cloud_from_depth, \
    get_reproject_offset, resize_image
from our_method.utils.scene_utils import compute_relative_cam_pose_from, align_model_pose, compute_object_z_offset, compute_object_z_offset_non_articulated, \
    compute_obj_bbox_info, align_obj_with_wall, get_vis_cam_trajectory
import our_method.utils.transform_utils as T

# Set of non-collidable categories
NON_COLLIDABLE_CATEGORIES = {
    "towel",
    "rug",
    "mirror",
    "picture",
    "painting",
    "window",
    "art",
}

CATEGORIES_MUST_ON_FLOOR = {
    "rug",
    "carpet"
}


class TaskSceneGenerator:
    """
    3rd Step in ACDC pipeline. This takes in the output from Step 2 (Digital Cousin Matching) and generates
    fully populated digital cousin scenes

    Foundation models used:
        - GPT-4O (https://openai.com/index/hello-gpt-4o/)
        - CLIP (https://github.com/openai/CLIP)
        - DINOv2 (https://github.com/facebookresearch/dinov2)

    Inputs:
        - Output from Step 2, which includes the following:
            - Per-object (category,, model, pose) digital cousin information

    Outputs:
        - Ordered digital cousin (category, model, pose) information per detected object from Step 1
    """
    SAMPLING_METHODS = {
        "random",
        "ordered",
    }

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
            task_feature_matching_path,
            gpt_api_key,
            gpt_version="4o",
            n_scenes=1,
            # sampling_method="random",
            sampling_method="ordered",
            resolve_collision=True,
            discard_objs=None,
            save_dir=None,
            visualize_scene=False,
            visualize_scene_tilt_angle=0,
            visualize_scene_radius=5,
            save_visualization=True
    ):
        """
        Runs the simulated scene generator. This does the following steps for all detected objects from Step and all
        matched cousin assets from Step 2:

        1. Compute camera pose and world origin point from step 1 output.
        2. Separately set each object in correct position and orientation w.r.t. the viewer camera,
           and save the relative transformation between the object and the camera.
        3. Put all objects in a single scene.
        4. Infer objects OnTop relationship. We currently only support OnTop cross-object relationship, so there might
            be artifacts if an object is 'In' another object, like books in a bookshelf.
        5. Process collisions and put objects onto the floor or objects beneath to generate a physically plausible scene.
        6. (Optionally) visualize the reconstructed scene.

        Args:
            step_1_output_path (str): Absolute path to the output file generated from Step 1 (RealWorldExtractor)
            step_2_output_path (str): Absolute path to the output file generated from Step 2 (DigitalCousinMatcher)
            n_scenes (int): Number of scenes to generate. This number cannot be greater than the number of cousins
                generated from Step 2 if @sampling_method="ordered" or greater than the product of all possible cousin
                combinations if @sampling_method="random"
            sampling_method (str): Sampling method to use when generating scenes. "random" will randomly select a cousin
                for each detected object in Step 1 (total combinations: N_cousins ^ N_objects). "ordered" will
                sequentially iterate over each detected object and generate scenes with corresponding ordered cousins,
                i.e.: a scene with all 1st cousins, a scene with all 2nd cousins, etc. (total combinations: N_cousins).
                Note that in both cases, the first scene generated will always be composed of all the closest (first)
                cousins. Default is "random"
            resolve_collision (bool): Whether to depenetrate collisions. When the point cloud is not denoised properly,
                or the mounting type is wrong, the object can be unreasonably large. Or when two objects in the input image
                intersect with each other, we may move an object by a non-trivial distance to depenetrate collision, so
                objects on top may fall down to the floor, and other objects may also need to be moved to avoid collision
                with this object. Under both cases, we recommend setting @resolve_collision to False to visualize the
                raw output.
            discard_objs (str): Names of objects to discard during reconstruction, seperated by comma, i.e., obj_1,obj_2,obj_3.
                Do not add space between object names.
            save_dir (None or str): If specified, the absolute path to the directory to save all generated outputs. If
                not specified, will generate a new directory in the same directory as @step_2_output_path
            visualize_scene (bool): Whether to visualize the scene after reconstruction. If True, the viewer camera will
                rotate around the scene's center point with a @visualize_scene_tilt_angle tilt cangle, and a 
                @visualize_scene_radius radius.
            visualize_scene_tilt_angle (float): The camera tilt angle in degree when visualizing the reconstructed scene. 
                This parameter is only used when @visualize_scene is set to True
            visualize_scene_radius (float): The camera rotating raiud in meters when visualizing the reconstructed scene.
                This parameter is only used when @visualize_scene is set to True
            save_visualization (bool): Whether to save the visualization results. This parameter is only used when 
                @visualize_scene is set to True

        Returns:
            2-tuple:
                bool: True if the process completed successfully. If successful, this will write all relevant outputs to
                    the directory specified in the second output
                None or str: If successful, this will be the absolute path to the main output file. Otherwise, None
        """
        ###### Step 2 결과 로드 ######
        if save_dir is None:
            save_dir = os.path.dirname(os.path.dirname(task_feature_matching_path))
        
        save_dir = os.path.join(save_dir, "task_scene_generation")
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Load step 2 info
        with open(step_3_output_path, "r") as f:
            step_3_output_info = json.load(f)
        


        # Launch omnigibson
        og.launch()
        
        # print(step_3_output_info['scene_0'])
        # print(step_3_output_info['scene_0'].keys())
        # exit()

        scene_info = step_3_output_info['scene_0']
        scene, cam_pose = TaskSceneGenerator.load_scene(scene_info=scene_info, visual_only=True)
        # obj = scene.object_registry("name", "cup_0")
        # print(obj._joints)
        # exit()
        print("cam_pose : ", cam_pose)

        scene_rgb = self.take_photo(n_render_steps=10)

        # scene_rgb = self.joint_test(scene, n_render_steps=10)
        # print(task_feature_matching_path)
        # print("#######################################################")
        # exit()
        with open(task_feature_matching_path, "r") as f:
            task_obj_output_info = json.load(f)
        
        
        # scene_rgb = self.take_photo(n_render_steps=100000)
        # TODO

        scene = TaskSceneGenerator.add_task_object(scene=scene, scene_info=scene_info, cam_pose=cam_pose, obj_info_json=task_obj_output_info, gpt_api_key=gpt_api_key, gpt_version=gpt_version, save_dir=save_dir, visual_only=True)
        # scene_rgb = self.take_photo(n_render_steps=1000000)
        # og.sim.viewer_camera.set_position_orientation(th.tensor([-0.3616, -2.9108,  2.2365], dtype=th.float), th.tensor([5.3933e-01, 6.4206e-03, 4.1202e-18, 8.4207e-01], dtype=th.float))
        # TODO
        scene_rgb = self.take_photo(n_render_steps=100)
        # exit(())


        cam_pose = scene_info["cam_pose"]
        # cam_pos = [0.15453, -4.16293, 2.24934]
        # cam_quat = [0.64139, -0.00021, 0.0, 0.76722]
        
        # cam_pos = [-0.71511, -2.44625, 1.38723]
        # cam_quat = [0.61408, -0.10552, -0.141, 0.76935]
        # Bed camera setting
        # cam_pos = [-0.35551, 0.54989, 4.3775]
        # cam_quat = [-0.01576, 0.00582, -0.02624, 0.99951]
        # og.sim.viewer_camera.set_position_orientation(th.tensor(cam_pos, dtype=th.float), th.tensor(cam_quat, dtype=th.float))
        og.sim.viewer_camera.set_position_orientation(th.tensor(cam_pose[0], dtype=th.float), th.tensor(cam_pose[1], dtype=th.float))
        
        print(og.sim.viewer_camera.get_position_orientation())
        # og.sim.viewer_camera.set_position_orientation(th.tensor([-0.3616, -2.9108,  2.2365], dtype=th.float), th.tensor([5.3933e-01, 6.4206e-03, 4.1202e-18, 8.4207e-01], dtype=th.float))
        
        scene_rgb = self.take_photo(n_render_steps=100)

        final_scene_info = deepcopy(scene_info)

        # Save final info
        with open(f"{save_dir}/scene_info.json", "w+") as f:
            json.dump(final_scene_info, f, indent=4, cls=NumpyTorchEncoder)
        print(final_scene_info)
        exit()


        return True, step_3_output_path

    
    @staticmethod
    def create_scene(floor=True, sky=True):
        """
        Helper function for creating new empty scene in OmniGibson

        Args:
            floor (bool): Whether to use floor or not
            sky (bool): Whether to use sky or not

        Returns:
            Scene: OmniGibson scene
        """
        og.sim.stop()
        og.clear()
        scene = Scene(use_floor_plane=floor, floor_plane_visible=floor, use_skybox=sky)
        og.sim.import_scene(scene)
        og.sim.play()
        return scene

    @staticmethod
    def load_scene(scene_info, visual_only=False):
        """
        Loads the cousin scene specified by info at @scene_info_fpath

        Args:
            scene_info (dict or str): If dict, scene information to load. Otherwise, should be absolute path to the
                scene info that should be loaded
            visual_only (bool): Whether to load all objects as visual only or not

        Returns:
            Scene: loaded OmniGibson scene
        """
        # Stop sim, clear it, then load empty scene
        scene = TaskSceneGenerator.create_scene(floor=True)

        # Load scene information if it's a path
        if isinstance(scene_info, str):
            with open(scene_info, "r") as f:
                scene_info = json.load(scene_info)

        # Set viewer camera to proper pose
        cam_pose = scene_info["cam_pose"]
        og.sim.viewer_camera.set_position_orientation(th.tensor(cam_pose[0], dtype=th.float), th.tensor(cam_pose[1], dtype=th.float))

        # Load all objects
        with og.sim.stopped():
            for obj_name, obj_info in scene_info["objects"].items():
                obj = DatasetObject(
                    name=obj_name,
                    category=obj_info["category"],
                    model=obj_info["model"],
                    visual_only=visual_only,
                    # kinematic_only=True, 
                    fixed_base=True,
                    scale=obj_info["scale"]
                )
                scene.add_object(obj)
                obj_pos, obj_quat = T.mat2pose(T.pose_in_A_to_pose_in_B(
                    pose_A=np.array(obj_info["tf_from_cam"]),
                    pose_A_in_B=T.pose2mat(cam_pose),
                ))
                obj.set_position_orientation(th.tensor(obj_pos, dtype=th.float), th.tensor(obj_quat, dtype=th.float))

        # Initialize all objects by taking one step
        og.sim.step()
        return scene, cam_pose

    def take_photo(self, n_render_steps=5):
        """
        Takes photo with current scene configuration with current camera

        Args:
            n_render_steps (int): Number of rendering steps to take before taking the photo

        Returns:
            np.ndarray: (H,W,3) RGB frame from viewer camera perspective
        """
        # Render a bit,
        for _ in range(n_render_steps):
            og.sim.render()
        rgb = og.sim.viewer_camera.get_obs()[0]["rgb"][:, :, :3].cpu().detach().numpy()
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).astype(np.uint8)

        img = Image.fromarray(rgb)
        img.save("./our_method_test/acdc_output/viewer_rgb.png")

        return rgb

    def joint_test(self, scene, n_render_steps=5):
        """
        Takes photo with current scene configuration with current camera

        Args:
            n_render_steps (int): Number of rendering steps to take before taking the photo

        Returns:
            np.ndarray: (H,W,3) RGB frame from viewer camera perspective
        """
        obj = scene.object_registry("name", "cabinet_0") 
        print(obj.joints.keys())  

        joint_index = 0
        positions = obj.get_joint_positions() # 기존 joint 위치 가져오기
        print("Initial positions:", positions)

        for step in range(n_render_steps):
            # 10번째 스텝마다 joint 위치 변경 (0 ↔ 1.5)
            if step % 50 == 0:
                positions[joint_index] = 1.5 if positions[joint_index] == 0 else 0
                obj.set_joint_positions(positions)  # 업데이트
                print(f"Step {step}: Updated joint position -> {positions[joint_index]}")

            # 물리 시뮬레이션 실행 및 렌더링
            og.sim.step_physics()
            og.sim.render()
        rgb = og.sim.viewer_camera.get_obs()[0]["rgb"][:, :, :3].cpu().detach().numpy()
        return rgb
    
    
    # def add_task_object(scene, scene_info, cam_pose, obj_info_json, gpt_api_key, gpt_version, save_dir, visual_only=False, probability_map = True, capture_env=False):
    #      # Load all objects
    #     assert gpt_api_key is not None, "gpt_api_key must be specified in order to use GPT model!"
    #     gpt = GPT(api_key=gpt_api_key, version=gpt_version, log_dir_tail="TaskSceneGeneration")
    #     object_retrieval_save_dir = os.path.join(os.path.dirname(save_dir), "task_object_retrieval")
    #     with og.sim.stopped():
    #         count = 1

    #         for obj_name, obj_info in obj_info_json["objects"].items():
    #             obj = DatasetObject(
    #                 name=obj_name,
    #                 category=obj_info["category"],
    #                 model=obj_info["model"],
    #                 visual_only=visual_only,
    #                 scale=np.array(obj_info["scale"])
    #             )
    #             scene.add_object(obj)
                
    #             child_obj_name = obj_name
    #             # "task_obj_cup"
    #             parent_obj_name = obj_info["parent_object"]
    #             placement = obj_info["placement"]
    #             sampled_point_3d = None
    #             probability_map_sampler = None

    #             if placement == "above":
    #                 from scipy.stats import multivariate_normal
    #                 import matplotlib.pyplot as plt

    #                 def find_model_image(model_id, image_dir):
    #                     """
    #                     모델 ID로 시작하는 PNG 이미지를 image_dir에서 찾아 경로 반환

    #                     Args:
    #                         model_id (str): 예: "lclkju"
    #                         image_dir (str): 이미지들이 저장된 디렉토리 경로

    #                     Returns:
    #                         str or None: 해당 이미지의 경로 (없으면 None)
    #                     """
    #                     for fname in os.listdir(image_dir):
    #                         if fname.startswith(model_id) and fname.endswith(".png"):
    #                             return os.path.join(image_dir, fname)
    #                     return None
                    
    #                 def compute_probability_map_on_plane(
    #                     bbox_points_world: np.ndarray,
    #                     center_points_world: np.ndarray,
    #                     grid_res: int = 500,
    #                     cov: float = 0.001,
    #                     weights: np.ndarray = None,
    #                 ):
    #                     """
    #                     4개의 bbox 꼭짓점으로 정의된 평면 위에,
    #                     9개의 중심점을 투영하여 확률 맵 생성.

    #                     Args:
    #                         bbox_points_world: (4, 3) ndarray, 3D 꼭짓점
    #                         center_points_world: (N, 3) ndarray, 3D 중심점들
    #                         grid_res: 확률 맵 해상도
    #                         cov: 각 가우시안 분포의 공분산 (float)
    #                         weights: (N,) 확률 가중치, 없으면 uniform

    #                     Returns:
    #                         prob_map: (grid_res, grid_res) ndarray, 확률 분포
    #                         center_points_2d: (N, 2) ndarray, 평면상 투영된 중심점 좌표
    #                         bounds: (u_min, u_max, v_min, v_max), 확률 맵 좌표 범위
    #                     """
    #                     # 1. 평면 좌표계 정의
    #                     P0, P1, _, P3 = bbox_points_world
    #                     origin = P0
    #                     x_axis = P1 - P0
    #                     x_axis /= np.linalg.norm(x_axis)
    #                     y_axis = P3 - P0
    #                     y_axis -= x_axis * np.dot(y_axis, x_axis)
    #                     y_axis /= np.linalg.norm(y_axis)

    #                     # 2. 3D → 평면 투영
    #                     def project_to_plane(p):
    #                         vec = p - origin
    #                         u = np.dot(vec, x_axis)
    #                         v = np.dot(vec, y_axis)
    #                         return np.array([u, v])

    #                     center_points_2d = np.array([project_to_plane(p) for p in center_points_world])
    #                     bbox_2d = np.array([project_to_plane(p) for p in bbox_points_world])
    #                     u_min, u_max = bbox_2d[:,0].min(), bbox_2d[:,0].max()
    #                     v_min, v_max = bbox_2d[:,1].min(), bbox_2d[:,1].max()



    #                     # # 3. 2D grid 설정
    #                     # u_min, u_max = center_points_2d[:, 0].min(), center_points_2d[:, 0].max()
    #                     # v_min, v_max = center_points_2d[:, 1].min(), center_points_2d[:, 1].max()
    #                     uu, vv = np.meshgrid(np.linspace(u_min, u_max, grid_res),
    #                                         np.linspace(v_min, v_max, grid_res))
    #                     grid_points = np.stack([uu, vv], axis=-1)

    #                     # 4. 확률맵 생성
    #                     if weights is None:
    #                         weights = np.ones(len(center_points_2d)) / len(center_points_2d)

    #                     prob_map = np.zeros((grid_res, grid_res))
    #                     for mu, w in zip(center_points_2d, weights):
    #                         rv = multivariate_normal(mean=mu, cov=cov)
    #                         prob_map += w * rv.pdf(grid_points)

    #                     prob_map /= prob_map.sum()

    #                     # return prob_map, center_points_2d, (u_min, u_max, v_min, v_max)
    #                     return prob_map, center_points_2d, (u_min, u_max, v_min, v_max), origin, x_axis, y_axis

    #                 blended_number_img_path, grid_centers_world, bbox_world = TaskSceneGenerator.get_VLM_prompt_image(scene, 
    #                                                                                   parent_obj_name=parent_obj_name, 
    #                                                                                   child_obj_name=child_obj_name,
    #                                                                                   parent_re_axis_mat=obj_info.get("parent_re_axis_mat", np.diag([1, 1, 1])),
    #                                                                                   child_re_axis_mat=obj_info.get("re_axis_mat", np.diag([1, 1, 1])),
    #                                                                                   output_dir=save_dir
    #                                                                                   )
    #                 front_parent_img_path = find_model_image(scene_info["objects"][parent_obj_name]["model"], 
    #                                                          f"{object_retrieval_save_dir}/{child_obj_name}/parent_object_front_pose_select/")
    #                 front_child_img_path = find_model_image(obj_info_json["objects"][child_obj_name]["model"], 
    #                                         f"{object_retrieval_save_dir}/{child_obj_name}/task_object_front_pose_select/")
                    
    #                 assert front_parent_img_path is not None or front_child_img_path is not None, \
    #                 f"❌ Image not found: Could not find a model image for either parent '{parent_obj_name}' or child '{child_obj_name}'."
                                        
    #                 def save_probability_map_image(prob_map, center_points_2d, bounds, save_path="./our_method_test/acdc_output/task_scene_generation/prob_map_vis.png"):
    #                     import matplotlib
    #                     matplotlib.use('Agg')  # GUI 없이 이미지 저장 전용 백엔드
    #                     u_min, u_max, v_min, v_max = bounds

    #                     plt.figure(figsize=(6, 6))
    #                     plt.imshow(prob_map, extent=[u_min, u_max, v_min, v_max], origin='lower', cmap='viridis')
    #                     plt.scatter(center_points_2d[:, 0], center_points_2d[:, 1], c='r', label='Centers')
    #                     plt.title("2D Probability Map on Projected Plane")
    #                     plt.colorbar(label='Probability Density')
    #                     plt.axis('equal')
    #                     plt.legend()
    #                     plt.tight_layout()

    #                     plt.savefig(save_path, dpi=300)
    #                     plt.close()
    #                     print(f"[✅] Saved probability map image to: {save_path}")

    #                 if probability_map:
    #                     nn_selection_payload = gpt.payload_above_object_distribution(
    #                             prompt_img_path = blended_number_img_path,
    #                             parent_obj_name = parent_obj_name,
    #                             placement = placement,
    #                             child_obj_name = child_obj_name,
    #                             parent_front_view_img_path = front_parent_img_path,
    #                             child_front_view_img_path = front_child_img_path)                    

    #                     gpt_text_response = gpt(nn_selection_payload)
    #                     print(f"gpt_text_response: {gpt_text_response}")
    #                     if gpt_text_response is None:
    #                         print(f"gpt_text_response is None")
    #                         # Failed, terminate early
    #                         return False, None

    #                     # gpt_text_response = "1: 0.15, 2: 0.20, 3: 0.05, 4: 0.15, 5: 0.00, 6: 0.00, 7: 0.15, 8: 0.20, 9: 0.10"
    #                     # gpt_text_response = "1: 0.10, 2: 0.00, 3: 0.00, 4: 0.10, 5: 0.20, 6: 0.20, 7: 0.00, 8: 0.20, 9: 0.20"


    #                     # 정규식으로 추출 후 dict로 변환
    #                     prob_dict = {
    #                         int(k): float(v)
    #                         for k, v in re.findall(r"(\d+):\s*([01]\.\d+)", gpt_text_response)
    #                     }

    #                     print(prob_dict)
    #                     # prob_map, center_points_2d, (u_min, u_max, v_min, v_max) = compute_probability_map_on_plane(bbox_world, grid_centers_world, cov=0.005, weights=prob_dict.values())
    #                     (prob_map, center_points_2d, bounds, origin, x_axis, y_axis) = compute_probability_map_on_plane(bbox_world, grid_centers_world, cov=0.005, weights=prob_dict.values())
    #                     save_probability_map_image(prob_map, center_points_2d, bounds)
    #                     # 샘플러 초기화
    #                     probability_map_sampler = ProbMapSampler(prob_map, bounds, origin, x_axis, y_axis)
    #                     obj_info_json["objects"][obj_name]["probability_map"] = probability_map_sampler

    #                     # 나중에 3D 좌표 1개 뽑기
    #                     sampled_point_3d = probability_map_sampler.sample()
                        
    #                     print(f"sampled_point_3d: {sampled_point_3d}")
    #                 else:
    #                     nn_selection_payload = gpt.payload_above_object_position(
    #                             prompt_img_path = blended_number_img_path,
    #                             parent_obj_name = parent_obj_name,
    #                             placement = placement,
    #                             child_obj_name = child_obj_name,
    #                             parent_front_view_img_path = front_parent_img_path,
    #                             child_front_view_img_path = front_child_img_path)                    

    #                     gpt_text_response = gpt(nn_selection_payload)
    #                     print(f"gpt_text_response: {gpt_text_response}")
    #                     if gpt_text_response is None:
    #                         print(f"gpt_text_response is None")
    #                         # Failed, terminate early
    #                         return False, None

    #                     # Extract the first non-negative integer from the response
    #                     match = re.search(r'\b\d+\b', gpt_text_response)

    #                     if match:
    #                         selected_index = int(match.group())
    #                         print(f"Selected index: {selected_index}")
    #                     else:
    #                         print("No valid number found.")
    #                         return False, None                 



    #                     probability_map_sampler = None
    #                     sampled_point_3d = grid_centers_world[selected_index-1]
    #                     print(f"sampled_point_3d: {sampled_point_3d}")                  

    #             obj_pos, obj_quat, child_clearance_axis, child_clearance_sign = TaskSceneGenerator.place_relative_to(count, scene, 
    #                                                                      child_obj_name=child_obj_name,
    #                                                                      parent_obj_name=parent_obj_name,
    #                                                                      placement=placement,
    #                                                                      parent_re_axis_mat=obj_info.get("parent_re_axis_mat", np.diag([1, 1, 1])),
    #                                                                      child_re_axis_mat=obj_info.get("re_axis_mat", np.diag([1, 1, 1])),
    #                                                                      above_sampled_point_3d=sampled_point_3d,
    #                                                                     )
    #             obj_info_json["objects"][obj_name]["clearance_axis"] = child_clearance_axis
    #             obj_info_json["objects"][obj_name]["clearance_sign"] = child_clearance_sign
    #             count += 1


    #             # obj_pos, obj_quat = T.mat2pose(T.pose_in_A_to_pose_in_B(
    #             #     pose_A=np.array(obj_info["tf_from_cam"]),
    #             #     pose_A_in_B=T.pose2mat(cam_pose),
    #             # ))
    #             # obj_pos = np.array([0,0,0])
    #             # obj_quat = np.array([0,0,0,1])
    #             # print("obj_pos: ", obj_pos)
    #             # print("obj_quat: ", obj_quat) #  X, Y, Z, W 순서

    #             # rel_tf = T.relative_pose_transform(obj_pos, obj_quat, cam_pose[0], cam_pose[1])
    #             # print(T.pose2mat(rel_tf))
                
    #             obj.set_position_orientation(th.tensor(obj_pos, dtype=th.float), th.tensor(obj_quat, dtype=th.float))
        

    #             # TODO
    #     og.sim.step()
    #     for obj_name, obj_info in obj_info_json["objects"].items():
    #         print("obj_name: ", obj_name)
    #         TaskSceneGenerator.snap_to_place(scene, 
    #                                         obj_info,
    #                                         cam_pose=cam_pose,
    #                                         child_obj_name=obj_name,
    #                                         parent_obj_name=obj_info["parent_object"],
    #                                         placement=obj_info["placement"])
            
    #         # TaskSceneGenerator.random_position_on_parent(scene, 
    #         #                                 obj_info,
    #         #                                 cam_pose=cam_pose,
    #         #                                 child_obj_name=obj_name,
    #         #                                 parent_obj_name=obj_info["parent_object"],
    #         #                                 parent_re_axis_mat=obj_info.get("parent_re_axis_mat", np.diag([1, 1, 1])),
    #         #                                 child_re_axis_mat=obj_info.get("re_axis_mat", np.diag([1, 1, 1])))
    #         if not capture_env :

    #         if probability_map:
    #             TaskSceneGenerator.above_collision_check(scene, 
    #                                             obj_info,
    #                                             child_obj_name=obj_name,
    #                                             parent_obj_name=obj_info["parent_object"],
    #                                             probability_map_sampler=obj_info_json["objects"][obj_name]["probability_map"],
    #                                             above_sampled_point_3d=sampled_point_3d)
                                            
            
    #         obj = scene.object_registry("name", obj_name)
    #         # TaskSceneGenerator.align_object_z_axis(scene, 
    #         #                                 obj_info,
    #         #                                 cam_pose=cam_pose,
    #         #                                 child_obj_name=obj_name)
    #          # 현재 child의 중심 위치
    #         obj_pos, obj_quat = obj.get_position_orientation()

    #         # 추가된 오브젝트 Scene Info 추가
    #         obj_scene_info = {
    #             "category": obj.category,
    #             "model": obj.model,
    #             "scale": obj.scale,
    #             "bbox_extent": obj.aabb_extent.cpu().detach().numpy(),
    #             "tf_from_cam": T.pose2mat(T.relative_pose_transform(        
    #                 obj_pos,
    #                 obj_quat,
    #                 cam_pose[0],
    #                 cam_pose[1],
    #         )),
    #             # "mount": obj_info["mount"],
    #             "mount": {
    #                 "floor": False,
    #                 "wall": False,
    #             },
    #         }
        
    #         scene_info["objects"][obj_name] = obj_scene_info
        
    #     # Initialize all objects by taking one step
    #     og.sim.step()
    #     return scene
    
    def add_task_object(scene, scene_info, cam_pose, obj_info_json, gpt_api_key, gpt_version, save_dir, visual_only=False, probability_map = True, capture_env=False):
         # Load all objects
        assert gpt_api_key is not None, "gpt_api_key must be specified in order to use GPT model!"
        gpt = GPT(api_key=gpt_api_key, version=gpt_version, log_dir_tail="TaskSceneGeneration")
        object_retrieval_save_dir = os.path.join(os.path.dirname(save_dir), "task_object_retrieval")
        
        for obj_name, obj_info in obj_info_json["objects"].items():
            count = 1

            with og.sim.stopped():    
                obj = DatasetObject(
                    name=obj_name,
                    category=obj_info["category"],
                    model=obj_info["model"],
                    visual_only=visual_only,
                    scale=np.array(obj_info["scale"])
                )
                scene.add_object(obj)
                
                child_obj_name = obj_name
                # "task_obj_cup"
                parent_obj_name = obj_info["parent_object"]
                placement = obj_info["placement"]
                sampled_point_3d = None
                probability_map_sampler = None

                if placement == "above":
                    from scipy.stats import multivariate_normal
                    import matplotlib.pyplot as plt

                    def find_model_image(model_id, image_dir):
                        """
                        모델 ID로 시작하는 PNG 이미지를 image_dir에서 찾아 경로 반환

                        Args:
                            model_id (str): 예: "lclkju"
                            image_dir (str): 이미지들이 저장된 디렉토리 경로

                        Returns:
                            str or None: 해당 이미지의 경로 (없으면 None)
                        """
                        for fname in os.listdir(image_dir):
                            if fname.startswith(model_id) and fname.endswith(".png"):
                                return os.path.join(image_dir, fname)
                        return None
                    
                    def compute_probability_map_on_plane(
                        bbox_points_world: np.ndarray,
                        center_points_world: np.ndarray,
                        all_object_info: dict,
                        grid_res: int = 500,
                        cov: float = 0.001,
                        weights: np.ndarray = None,
                    ):
                        """
                        4개의 bbox 꼭짓점으로 정의된 평면 위에,
                        9개의 중심점을 투영하여 확률 맵 생성.

                        Args:
                            bbox_points_world: (4, 3) ndarray, 3D 꼭짓점
                            center_points_world: (N, 3) ndarray, 3D 중심점들
                            grid_res: 확률 맵 해상도
                            cov: 각 가우시안 분포의 공분산 (float)
                            weights: (N,) 확률 가중치, 없으면 uniform

                        Returns:
                            prob_map: (grid_res, grid_res) ndarray, 확률 분포
                            center_points_2d: (N, 2) ndarray, 평면상 투영된 중심점 좌표
                            bounds: (u_min, u_max, v_min, v_max), 확률 맵 좌표 범위
                        """
                        # print("##########################################################")
                        # print(all_object_info)
                        # exit()

                        # 1. 평면 좌표계 정의
                        P0, P1, _, P3 = bbox_points_world
                        origin = P0
                        x_axis = P1 - P0
                        x_axis /= np.linalg.norm(x_axis)
                        y_axis = P3 - P0
                        y_axis -= x_axis * np.dot(y_axis, x_axis)
                        y_axis /= np.linalg.norm(y_axis)

                        # 2. 3D → 평면 투영
                        def project_to_plane(p):
                            vec = p - origin
                            u = np.dot(vec, x_axis)
                            v = np.dot(vec, y_axis)
                            return np.array([u, v])

                        center_points_2d = np.array([project_to_plane(p) for p in center_points_world])
                        bbox_2d = np.array([project_to_plane(p) for p in bbox_points_world])
                        u_min, u_max = bbox_2d[:,0].min(), bbox_2d[:,0].max()
                        v_min, v_max = bbox_2d[:,1].min(), bbox_2d[:,1].max()



                        # # 3. 2D grid 설정
                        # u_min, u_max = center_points_2d[:, 0].min(), center_points_2d[:, 0].max()
                        # v_min, v_max = center_points_2d[:, 1].min(), center_points_2d[:, 1].max()
                        uu, vv = np.meshgrid(np.linspace(u_min, u_max, grid_res),
                                            np.linspace(v_min, v_max, grid_res))
                        grid_points = np.stack([uu, vv], axis=-1)

                        # 4. 확률맵 생성
                        if weights is None:
                            weights = np.ones(len(center_points_2d)) / len(center_points_2d)

                        prob_map = np.zeros((grid_res, grid_res))
                        for mu, w in zip(center_points_2d, weights):
                            rv = multivariate_normal(mean=mu, cov=cov)
                            prob_map += w * rv.pdf(grid_points)

                        prob_map /= prob_map.sum()

                        # return prob_map, center_points_2d, (u_min, u_max, v_min, v_max)
                        return prob_map, center_points_2d, (u_min, u_max, v_min, v_max), origin, x_axis, y_axis

                    import matplotlib
                    from matplotlib.path import Path as MplPath
                    from scipy.ndimage import binary_dilation
                    from shapely.geometry import Polygon

                    matplotlib.use('Agg')  # Headless 환경에서도 작동
                    def generate_prob_map_from_parent_bbox(
                        all_object_info: dict,
                        grid_res: int = 500,
                        prob_map_save_path: str = "./output/prob_map_vis.png"
                    ):
                        parent_bbox = all_object_info['parent_obj']['bbox_world']

                        # 좌표계 정의
                        x_axis = parent_bbox[1] - parent_bbox[0]
                        y_axis = parent_bbox[3] - parent_bbox[0]
                        origin = np.mean(parent_bbox, axis=0)

                        x_axis /= np.linalg.norm(x_axis)
                        y_axis = np.cross(np.cross(x_axis, y_axis), x_axis)
                        y_axis /= np.linalg.norm(y_axis)
                        z_axis = np.cross(x_axis, y_axis)

                        R = np.stack([x_axis, y_axis, z_axis], axis=1)

                        # parent_obj의 bbox 꼭짓점 평면 투영
                        parent_corners_2d = np.array([(R.T @ (pt - origin))[:2] for pt in parent_bbox])

                        # 평면상 그리드 설정
                        min_x, max_x = parent_corners_2d[:, 0].min(), parent_corners_2d[:, 0].max()
                        min_y, max_y = parent_corners_2d[:, 1].min(), parent_corners_2d[:, 1].max()
                        xx, yy = np.meshgrid(np.linspace(min_x, max_x, grid_res),
                                            np.linspace(min_y, max_y, grid_res))
                        grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)

                        # 초기 확률맵: parent polygon 내부를 1로 채움
                        polygon_path = MplPath(parent_corners_2d)
                        mask = polygon_path.contains_points(grid_points)
                        prob_map = mask.astype(np.float32).reshape((grid_res, grid_res))

                        # ✅ 먼저 child_radius 계산
                        child_bbox = all_object_info['child_obj']['bbox_world']
                        child_corners_2d = np.array([(R.T @ (pt - origin))[:2] for pt in child_bbox])
                        child_center = np.mean(child_corners_2d, axis=0)
                        dists = np.linalg.norm(child_corners_2d - child_center, axis=1)
                        child_radius = np.max(dists)
                        # child_radius = 0.05

                        # ✳️ 각 object별로 마스킹 + 실공간 거리 기반 확장 처리
                        for obj_name, obj_info in all_object_info.items():
                            if obj_name == "parent_obj" or "bbox_world" not in obj_info:
                                continue

                            obj_bbox = obj_info['bbox_world']
                            obj_corners_2d = np.array([(R.T @ (pt - origin))[:2] for pt in obj_bbox])

                            # polygon 확장 (meter 단위)
                            poly = Polygon(obj_corners_2d)
                            expanded_poly = poly.buffer(child_radius)

                            # 확장된 polygon을 기반으로 마스크 생성
                            expanded_path = MplPath(np.array(expanded_poly.exterior.coords))
                            obj_mask = expanded_path.contains_points(grid_points).reshape((grid_res, grid_res))

                            # 확률맵에서 해당 영역을 0으로 설정
                            prob_map[obj_mask] = 0.0


                        # 시각화 및 저장
                        os.makedirs(os.path.dirname(prob_map_save_path), exist_ok=True)
                        plt.figure(figsize=(6, 6))
                        plt.imshow(prob_map, extent=[min_x, max_x, min_y, max_y], origin='lower', cmap='viridis')
                        plt.title("Probability Map (masked by existing objects)")
                        plt.colorbar(label='Probability')
                        plt.axis('equal')
                        plt.tight_layout()
                        plt.savefig(prob_map_save_path, dpi=300)
                        plt.close()

                        print(f"[✅] Saved probability map image to: {prob_map_save_path}")

                        return prob_map

                    # def generate_prob_map_from_parent_bbox(
                    #     all_object_info: dict,
                    #     grid_res: int = 500,
                    #     prob_map_save_path: str = "./output/prob_map_vis.png"
                    # ):
                    #     parent_bbox = all_object_info['parent_obj']['bbox_world']

                    #     # 좌표계 정의
                    #     x_axis = parent_bbox[1] - parent_bbox[0]
                    #     y_axis = parent_bbox[3] - parent_bbox[0]
                    #     origin = np.mean(parent_bbox, axis=0)

                    #     x_axis /= np.linalg.norm(x_axis)
                    #     y_axis = np.cross(np.cross(x_axis, y_axis), x_axis)
                    #     y_axis /= np.linalg.norm(y_axis)
                    #     z_axis = np.cross(x_axis, y_axis)

                    #     R = np.stack([x_axis, y_axis, z_axis], axis=1)

                    #     # parent_obj의 bbox 꼭짓점 평면 투영
                    #     parent_corners_2d = np.array([(R.T @ (pt - origin))[:2] for pt in parent_bbox])

                    #     # 평면상 그리드 설정
                    #     min_x, max_x = parent_corners_2d[:, 0].min(), parent_corners_2d[:, 0].max()
                    #     min_y, max_y = parent_corners_2d[:, 1].min(), parent_corners_2d[:, 1].max()
                    #     xx, yy = np.meshgrid(np.linspace(min_x, max_x, grid_res),
                    #                         np.linspace(min_y, max_y, grid_res))
                    #     grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)

                    #     # 초기 확률맵: parent polygon 내부를 1로 채움
                    #     polygon_path = MplPath(parent_corners_2d)
                    #     mask = polygon_path.contains_points(grid_points)
                    #     prob_map = mask.astype(np.float32).reshape((grid_res, grid_res))

                    #     # ✳️ 각 object의 bbox를 2D로 투영 후 해당 영역만 0으로 마스킹
                    #     for obj_name, obj_info in all_object_info.items():
                    #         if obj_name == "parent_obj" or "bbox_world" not in obj_info:
                    #             continue

                    #         obj_bbox = obj_info['bbox_world']
                    #         obj_corners_2d = np.array([(R.T @ (pt - origin))[:2] for pt in obj_bbox])
                    #         obj_path = MplPath(obj_corners_2d)
                    #         obj_mask = obj_path.contains_points(grid_points)
                    #         obj_mask = obj_mask.reshape((grid_res, grid_res))

                    #         # parent 내에서 object가 겹치는 영역은 0으로 덮어쓰기
                    #         prob_map[obj_mask] = 0.0

                        
                    #     # ✅ child_obj polygon → parent 평면 중심에 놓기
                    #     child_bbox = all_object_info['child_obj']['bbox_world']
                    #     child_corners_2d = np.array([(R.T @ (pt - origin))[:2] for pt in child_bbox])
                    #     child_center = np.mean(child_corners_2d, axis=0)  # 보통 [0, 0]이지만 일반화된 코드

                    #     # 각 꼭짓점에서 중심까지의 거리
                    #     dists = np.linalg.norm(child_corners_2d - child_center, axis=1)

                    #     # 최대 거리
                    #     child_radius = np.max(dists)

                    #     print(prob_map)
                    #     print(prob_map.shape)
                    #     # 시각화 및 저장
                    #     os.makedirs(os.path.dirname(prob_map_save_path), exist_ok=True)
                    #     plt.figure(figsize=(6, 6))
                    #     plt.imshow(prob_map, extent=[min_x, max_x, min_y, max_y], origin='lower', cmap='viridis')
                    #     plt.title("Probability Map (masked by existing objects)")
                    #     plt.colorbar(label='Probability')
                    #     plt.axis('equal')
                    #     plt.tight_layout()
                    #     plt.savefig(prob_map_save_path, dpi=300)
                    #     plt.close()

                    #     print(f"[✅] Saved probability map image to: {prob_map_save_path}")




                    blended_number_img_path, grid_centers_world, bbox_world, all_object_info = TaskSceneGenerator.get_VLM_prompt_image(scene, 
                                                                                      parent_obj_name=parent_obj_name, 
                                                                                      child_obj_name=child_obj_name,
                                                                                      parent_re_axis_mat=obj_info.get("parent_re_axis_mat", np.diag([1, 1, 1])),
                                                                                      child_re_axis_mat=obj_info.get("re_axis_mat", np.diag([1, 1, 1])),
                                                                                      output_dir=save_dir
                                                                                      )
                    front_parent_img_path = find_model_image(scene_info["objects"][parent_obj_name]["model"], 
                                                             f"{object_retrieval_save_dir}/{child_obj_name}/parent_object_front_pose_select/")
                    front_child_img_path = find_model_image(obj_info_json["objects"][child_obj_name]["model"], 
                                            f"{object_retrieval_save_dir}/{child_obj_name}/task_object_front_pose_select/")
                    
                    assert front_parent_img_path is not None or front_child_img_path is not None, \
                    f"❌ Image not found: Could not find a model image for either parent '{parent_obj_name}' or child '{child_obj_name}'."
                                        
                    def save_probability_map_image(prob_map, center_points_2d, bounds, save_path="./our_method_test/acdc_output/task_scene_generation/prob_map_vis.png"):
                        import matplotlib
                        matplotlib.use('Agg')  # GUI 없이 이미지 저장 전용 백엔드
                        u_min, u_max, v_min, v_max = bounds

                        plt.figure(figsize=(6, 6))
                        plt.imshow(prob_map, extent=[u_min, u_max, v_min, v_max], origin='lower', cmap='viridis')
                        plt.scatter(center_points_2d[:, 0], center_points_2d[:, 1], c='r', label='Centers')
                        plt.title("2D Probability Map on Projected Plane")
                        plt.colorbar(label='Probability Density')
                        plt.axis('equal')
                        plt.legend()
                        plt.tight_layout()

                        plt.savefig(save_path, dpi=300)
                        plt.close()
                        print(f"[✅] Saved probability map image to: {save_path}")

                    if probability_map:
                        nn_selection_payload = gpt.payload_above_object_distribution(
                                prompt_img_path = blended_number_img_path,
                                parent_obj_name = parent_obj_name,
                                placement = placement,
                                child_obj_name = child_obj_name,
                                parent_front_view_img_path = front_parent_img_path,
                                child_front_view_img_path = front_child_img_path)                    

                        gpt_text_response = gpt(nn_selection_payload)
                        print(f"gpt_text_response: {gpt_text_response}")
                        if gpt_text_response is None:
                            print(f"gpt_text_response is None")
                            # Failed, terminate early
                            return False, None

                        # gpt_text_response = "1: 0.15, 2: 0.20, 3: 0.05, 4: 0.15, 5: 0.00, 6: 0.00, 7: 0.15, 8: 0.20, 9: 0.10"
                        # gpt_text_response = "1: 0.00, 2: 0.00, 3: 0.00, 4: 0.15, 5: 0.25, 6: 0.15, 7: 0.15, 8: 0.20, 9: 0.10"


                        # 정규식으로 추출 후 dict로 변환
                        prob_dict = {
                            int(k): float(v)
                            for k, v in re.findall(r"(\d+):\s*([01]\.\d+)", gpt_text_response)
                        }
                        # print(prob_dict)

                        # prob_map, center_points_2d, (u_min, u_max, v_min, v_max) = compute_probability_map_on_plane(bbox_world, grid_centers_world, cov=0.005, weights=prob_dict.values())
                        prob_map_mask = generate_prob_map_from_parent_bbox(all_object_info, prob_map_save_path="./our_method_test/acdc_output/task_scene_generation/figure.png")
                        
                        # TODO
                        (prob_map, center_points_2d, bounds, origin, x_axis, y_axis) = compute_probability_map_on_plane(bbox_world, grid_centers_world, all_object_info, cov=0.005, weights=prob_dict.values())
                        prob_map *= prob_map_mask
                        save_probability_map_image(prob_map, center_points_2d, bounds)
                        # exit()
                        # 샘플러 초기화
                        probability_map_sampler = ProbMapSampler(prob_map, bounds, origin, x_axis, y_axis)
                        obj_info_json["objects"][obj_name]["probability_map"] = probability_map_sampler

                        # 나중에 3D 좌표 1개 뽑기
                        sampled_point_3d = probability_map_sampler.sample()
                        
                        print(f"sampled_point_3d: {sampled_point_3d}")
                    else:
                        nn_selection_payload = gpt.payload_above_object_position(
                                prompt_img_path = blended_number_img_path,
                                parent_obj_name = parent_obj_name,
                                placement = placement,
                                child_obj_name = child_obj_name,
                                parent_front_view_img_path = front_parent_img_path,
                                child_front_view_img_path = front_child_img_path)                    

                        gpt_text_response = gpt(nn_selection_payload)
                        print(f"gpt_text_response: {gpt_text_response}")
                        if gpt_text_response is None:
                            print(f"gpt_text_response is None")
                            # Failed, terminate early
                            return False, None

                        # Extract the first non-negative integer from the response
                        match = re.search(r'\b\d+\b', gpt_text_response)

                        if match:
                            selected_index = int(match.group())
                            print(f"Selected index: {selected_index}")
                        else:
                            print("No valid number found.")
                            return False, None                 



                        probability_map_sampler = None
                        sampled_point_3d = grid_centers_world[selected_index-1]
                        print(f"sampled_point_3d: {sampled_point_3d}")                  

                obj_pos, obj_quat, child_clearance_axis, child_clearance_sign = TaskSceneGenerator.place_relative_to(count, scene, 
                                                                         child_obj_name=child_obj_name,
                                                                         parent_obj_name=parent_obj_name,
                                                                         placement=placement,
                                                                         parent_re_axis_mat=obj_info.get("parent_re_axis_mat", np.diag([1, 1, 1])),
                                                                         child_re_axis_mat=obj_info.get("re_axis_mat", np.diag([1, 1, 1])),
                                                                         above_sampled_point_3d=sampled_point_3d,
                                                                        )
                obj_info_json["objects"][obj_name]["clearance_axis"] = child_clearance_axis
                obj_info_json["objects"][obj_name]["clearance_sign"] = child_clearance_sign
                count += 1


                # obj_pos, obj_quat = T.mat2pose(T.pose_in_A_to_pose_in_B(
                #     pose_A=np.array(obj_info["tf_from_cam"]),
                #     pose_A_in_B=T.pose2mat(cam_pose),
                # ))
                # obj_pos = np.array([0,0,0])
                # obj_quat = np.array([0,0,0,1])
                # print("obj_pos: ", obj_pos)
                # print("obj_quat: ", obj_quat) #  X, Y, Z, W 순서

                # rel_tf = T.relative_pose_transform(obj_pos, obj_quat, cam_pose[0], cam_pose[1])
                # print(T.pose2mat(rel_tf))
                
                obj.set_position_orientation(th.tensor(obj_pos, dtype=th.float), th.tensor(obj_quat, dtype=th.float))
        

                # TODO
            og.sim.step()
        
            print("obj_name: ", obj_name)
            TaskSceneGenerator.snap_to_place(scene, 
                                            obj_info,
                                            cam_pose=cam_pose,
                                            child_obj_name=obj_name,
                                            parent_obj_name=obj_info["parent_object"],
                                            placement=obj_info["placement"])
            
            # TaskSceneGenerator.random_position_on_parent(scene, 
            #                                 obj_info,
            #                                 cam_pose=cam_pose,
            #                                 child_obj_name=obj_name,
            #                                 parent_obj_name=obj_info["parent_object"],
            #                                 parent_re_axis_mat=obj_info.get("parent_re_axis_mat", np.diag([1, 1, 1])),
            #                                 child_re_axis_mat=obj_info.get("re_axis_mat", np.diag([1, 1, 1])))
            
            if probability_map:
                TaskSceneGenerator.above_collision_check(scene, 
                                                obj_info,
                                                child_obj_name=obj_name,
                                                parent_obj_name=obj_info["parent_object"],
                                                probability_map_sampler=obj_info_json["objects"][obj_name]["probability_map"],
                                                above_sampled_point_3d=sampled_point_3d)
                # TaskSceneGenerator.snap_to_place(scene, 
                #                             obj_info,
                #                             cam_pose=cam_pose,
                #                             child_obj_name=obj_name,
                #                             parent_obj_name=obj_info["parent_object"],
                #                             placement=obj_info["placement"])
                TaskSceneGenerator.enable_collision_and_physics(scene, child_obj_name=obj_name, parent_obj_name=obj_info["parent_object"])
                                            
            
            obj = scene.object_registry("name", obj_name)
            # TaskSceneGenerator.align_object_z_axis(scene, 
            #                                 obj_info,
            #                                 cam_pose=cam_pose,
            #                                 child_obj_name=obj_name)
             # 현재 child의 중심 위치
            obj_pos, obj_quat = obj.get_position_orientation()

            # 추가된 오브젝트 Scene Info 추가
            obj_scene_info = {
                "category": obj.category,
                "model": obj.model,
                "scale": obj.scale,
                "bbox_extent": obj.aabb_extent.cpu().detach().numpy(),
                "tf_from_cam": T.pose2mat(T.relative_pose_transform(        
                    obj_pos,
                    obj_quat,
                    cam_pose[0],
                    cam_pose[1],
            )),
                # "mount": obj_info["mount"],
                "mount": {
                    "floor": False,
                    "wall": False,
                },
            }
        
            scene_info["objects"][obj_name] = obj_scene_info
        
        # Initialize all objects by taking one step
        og.sim.step()
        return scene
    
    @staticmethod
    def get_VLM_prompt_image(scene, parent_obj_name, child_obj_name, parent_re_axis_mat, child_re_axis_mat, output_dir = "./our_method_test/acdc_output/"):
        """
        위에서 본 parent object의 segmentation mask를 시각화하고 반환합니다.

        Args:
            scene: 현재 OmniGibson scene
            parent_obj_name (str): 시각화할 parent object 이름

        Returns:
            mask (np.ndarray): binary mask image (numpy array)
        """
        import matplotlib.pyplot as plt
        from omnigibson.sensors import VisionSensor
        import omni.replicator.core as rep
        import matplotlib.cm as cm
        from omnigibson.utils.transform_utils import pose2mat
        from scipy.spatial.transform import Rotation as R
        from PIL import Image, ImageDraw, ImageFont
        import torch
        import numpy as np
        import math
        import cv2
        # 미리 정의된 20가지 RGB 컬러 (0~255 범위)
        PRESET_COLORS = (np.array(plt.get_cmap("tab20").colors) * 255).astype(np.uint8)
        os.makedirs(output_dir, exist_ok=True)
        file_prefix = f"{parent_obj_name}_above_{child_obj_name}"


        def assign_colors_to_ids(id_to_label):
            ids = sorted(id_to_label.keys())
            color_map = {}
            for i, sid in enumerate(ids):
                color_map[sid] = tuple(PRESET_COLORS[i % len(PRESET_COLORS)])
            return color_map

        def render_segmentation_overlay(seg_tensor, id_to_label):
            seg_np = seg_tensor.cpu().numpy()
            H, W = seg_np.shape
            color_map = assign_colors_to_ids(id_to_label)

            seg_rgb = np.zeros((H, W, 3), dtype=np.uint8)
            for sid, color in color_map.items():
                seg_rgb[seg_np == sid] = color
            return seg_rgb
        
        def split_bbox_into_grid(bbox_corners, num_rows, num_cols):
            """
            4개의 꼭짓점(bbox의 bottom face 기준)으로 구성된 BBox를 x-y 평면에서 num_rows x num_cols grid로 나눔
            Returns 각 cell 중심점의 (x, y, z)
            """
            # 꼭짓점에서 x/y 범위 추출
            x_coords = bbox_corners[:, 0]
            y_coords = bbox_corners[:, 1]
            z_val = bbox_corners[0, 2]  # z는 일정하므로 하나만

            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()

            xs = np.linspace(x_min, x_max, num=num_cols + 1)
            ys = np.linspace(y_min, y_max, num=num_rows + 1)

            centers = []
            for r in range(num_rows):
                for c in range(num_cols):
                    cx = (xs[c] + xs[c+1]) / 2
                    cy = (ys[r] + ys[r+1]) / 2
                    centers.append([cx, cy, z_val])

            return np.array(centers)  # shape: (num_rows*num_cols, 3)
        
        def world_to_image(cam, points_world):
            from scipy.spatial.transform import Rotation as R
            def compute_intrinsics(cam):
                # 카메라 파라미터
                focal_length = cam.focal_length  # 초점 거리 (mm)
                width, height = cam.image_width, cam.image_height  # 이미지 해상도 (픽셀)
                horizontal_aperture = cam.horizontal_aperture  # 수평 아퍼처 (mm)

                # 1. 수평 시야각 (horizontal FOV) 계산
                horizontal_fov = 2 * math.atan(horizontal_aperture / (2 * focal_length))  # radian 단위

                # 2. 수직 시야각 (vertical FOV) 계산
                vertical_fov = horizontal_fov * height / width  # 수직 시야각

                # 3. fx, fy 계산
                fx = (width / 2.0) / math.tan(horizontal_fov / 2.0)  # 수평 초점 거리 (픽셀)
                fy = (height / 2.0) / math.tan(vertical_fov / 2.0)  # 수직 초점 거리 (픽셀)

                # 4. cx, cy 계산 (이미지의 중심)
                cx = width / 2
                cy = height / 2

                # 5. 내적 행렬 (Intrinsic Matrix) 계산
                K = th.tensor([
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0]
                ], dtype=th.float32)

                # print("========== Intrinsics ==========")
                # print("fx, fy:", fx, fy)
                # print("cx, cy:", cx, cy)
                # print("K:\n", K)
                # print("================================\n")

                return K

            # 1. 카메라 pose 가져오기
            cam_pos, cam_quat = cam.get_position_orientation()  # cam_quat: (x, y, z, w)
            cam_pos = torch.tensor(cam_pos, dtype=torch.float32)
            cam_quat = torch.tensor(cam_quat, dtype=torch.float32)
            # print("****************************************************")
            # print("cam_pos: ", cam_pos)
            # print("cam_quat: ", cam_quat)
            # print("****************************************************")


            # 2. pose2mat: 카메라 월드 좌표계 → 4x4 변환 행렬
            def pose2mat(pos, quat):
                r = R.from_quat(quat.numpy())  # (x, y, z, w)
                rot_mat = torch.tensor(r.as_matrix(), dtype=torch.float32)
                T = torch.eye(4, dtype=torch.float32)
                T[:3, :3] = rot_mat
                T[:3, 3] = pos
                return T

            cam_T_world = pose2mat(cam_pos, cam_quat)          # (4, 4)
            world_T_cam = torch.linalg.inv(cam_T_world)        # (4, 4)

            # # 3. Intrinsics 계산 (조정된 fx, fy 값)
            # fx, fy = 500, 500  # fx, fy 값을 적절한 값으로 조정
            # cx, cy = cam.image_width / 2, cam.image_height / 2
            # print("************ Camera Transform ************")
            # print("cam_T_world:\n", cam_T_world)
            # print("world_T_cam:\n", world_T_cam)
            # print("*******************************************\n")

            K = compute_intrinsics(cam)

            # 4. 변환 시작
            pixel_coords = []
            for idx, pt in enumerate(points_world):

                pt_world = torch.tensor(pt, dtype=torch.float32)
                pt_homo = torch.cat([pt_world, torch.tensor([1.0])])  # (4,)
                pt_cam = (world_T_cam @ pt_homo)[:3]                  # (x, y, z)

                pt_cam = torch.tensor([pt_cam[0], -pt_cam[1], -pt_cam[2]])

                # print(f"[{idx}] pt_world: {pt_world.tolist()}")
                # print(f"[{idx}] pt_cam: {pt_cam.tolist()} (before flip)")

                # # OpenCV 기준: z > 0 이어야 보임
                # if pt_cam[2] <= 1e-5:
                #     pixel_coords.append((-1, -1))  # 매우 작은 z 값을 가진 점을 처리
                #     continue

                # 5. Project to image plane
                uvw = K @ pt_cam
                u = int(uvw[0].item() / uvw[2].item())
                v = int(uvw[1].item() / uvw[2].item())


                # print(f"[{idx}] uvw: {uvw.tolist()}")
                # print(f"[{idx}] pixel: ({u}, {v})\n")
            
                # 6. 이미지 범위 내에 있는지 확인
                if 0 <= u < cam.image_width and 0 <= v < cam.image_height:
                    pixel_coords.append((u, v))
                else:
                    print(f"[{idx}] ❌ Out of bounds: ({u}, {v})\n")
                    pixel_coords.append((-1, -1))

            return pixel_coords
        
        def draw_numbers_on_blended_image(blended: Image.Image, pixel_coords: list) -> Image.Image:
            """
            blended: PIL.Image (RGBA)
            pixel_coords: list of (u, v)
            Returns: PIL.Image (RGBA) with white text on solid black rectangle background
            """
            blended_cv_rgba = np.array(blended)
            blended_cv_bgra = cv2.cvtColor(blended_cv_rgba, cv2.COLOR_RGBA2BGRA)

            height, width = blended_cv_bgra.shape[:2]
            base_scale = min(width, height) / 512  # 기준 사이즈 512x512 기준

            font_scale = 0.4 * base_scale
            thickness = max(int(1 * base_scale), 1)

            for i, (u, v) in enumerate(pixel_coords):
                if 0 <= u < width and 0 <= v < height:
                    label = f"[{i+1}]"

                    (text_w, text_h), baseline = cv2.getTextSize(
                        label,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=font_scale,
                        thickness=thickness
                    )

                    x1 = u - text_w // 2 - 4
                    y1 = v - text_h - 4
                    x2 = u + text_w // 2 + 4
                    y2 = v + baseline + 4

                    # ✅ 불투명한 검정색 배경 (BGRA = 0, 0, 0, 255)
                    cv2.rectangle(
                        blended_cv_bgra,
                        (x1, y1), (x2, y2),
                        color=(0, 0, 0, 255),
                        thickness=-1
                    )

                    print((u - text_w // 2, v))
                    # ✅ 흰색 텍스트 (불투명)
                    cv2.putText(
                        blended_cv_bgra,
                        label,
                        (u - text_w // 2, v),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=font_scale,
                        color=(255, 255, 255, 255),
                        thickness=thickness,
                        lineType=cv2.LINE_AA
                    )

            blended_with_numbers = Image.fromarray(cv2.cvtColor(blended_cv_bgra, cv2.COLOR_BGRA2RGBA))
            return blended_with_numbers
        
        def look_at_rotation(forward, up):
            # Normalize input vectors
            forward = forward / np.linalg.norm(forward)
            right = np.cross(up, forward)
            right = right / np.linalg.norm(right)
            true_up = np.cross(forward, right)
            
            rot_mat = np.stack([right, true_up, forward], axis=1)  # (3, 3)
            return rot_mat
        
        def transform_bbox_to_world(bbox_local: np.ndarray, position: np.ndarray, quaternion: np.ndarray):
            """
            bbox_local: (N, 3) in local frame (numpy array)
            position: (3,) numpy array
            quaternion: (4,) numpy array (x, y, z, w) format
            returns: (N, 3) numpy array in world frame
            """
            # 1. pose → transformation matrix
            pos_th = th.tensor(position, dtype=th.float32)
            quat_th = th.tensor(quaternion, dtype=th.float32)
            T_world = pose2mat((pos_th, quat_th))  # (4, 4)

            # 2. bbox → homogeneous coords
            bbox_local_th = th.tensor(bbox_local, dtype=th.float32)  # (N, 3)
            ones = th.ones((bbox_local_th.shape[0], 1))
            bbox_local_homo = th.cat([bbox_local_th, ones], dim=1)  # (N, 4)

            # 3. transform
            bbox_world_th = (T_world @ bbox_local_homo.T).T[:, :3]  # (N, 3)

            return bbox_world_th.numpy()

        if not isinstance(parent_re_axis_mat, np.ndarray):
            parent_re_axis_mat = np.array(parent_re_axis_mat)
        if not isinstance(child_re_axis_mat, np.ndarray):
            child_re_axis_mat = np.array(child_re_axis_mat)
        
        # 1. parent object 가져오기
        parent_obj = scene.object_registry("name", parent_obj_name)
        parent_pos, parent_quat = parent_obj.get_position_orientation()
        parent_pos = parent_pos.cpu().numpy() if isinstance(parent_pos, th.Tensor) else parent_pos
        parent_quat = parent_quat.cpu().numpy() if isinstance(parent_quat, th.Tensor) else parent_quat

        # parent의 회전 행렬
        parent_rot_mat = T.quat2mat(parent_quat)  # (3, 3)


        # 2. child object 가져오기
        child_obj = scene.object_registry("name", child_obj_name)
        child_pos, child_quat = child_obj.get_position_orientation()
        child_pos = child_pos.cpu().numpy() if isinstance(child_pos, th.Tensor) else child_pos
        child_quat = child_quat.cpu().numpy() if isinstance(child_quat, th.Tensor) else child_quat

        # 3. BBox 정보 추출
        parent_bbox_dict = compute_obj_bbox_info(parent_obj)
        child_bbox_dict = compute_obj_bbox_info(child_obj)
        parent_bbox = parent_bbox_dict["bbox_top_in_desired_frame"]
        child_bbox_raw = child_bbox_dict["bbox_top_in_desired_frame"]

        # 5. child bbox를 child 기준 → parent 기준 회전 변환
        child_bbox = (np.linalg.inv(child_re_axis_mat) @ child_bbox_raw.T).T  # local align
        R_rel = parent_re_axis_mat.T @ child_re_axis_mat
        vec = R_rel @ np.array([1, 0, 0])  # child의 x축이 parent 기준으로 얼마나 회전했는지

        # 6. child의 projected extent 계산
        child_axis = np.argmax(np.abs(vec))  # child의 주축
        if not child_axis:  # child x축이 parent x축과 정렬
            x_extent_c = (np.max(child_bbox[:, 0]) - np.min(child_bbox[:, 0])) / 2
            y_extent_c = (np.max(child_bbox[:, 1]) - np.min(child_bbox[:, 1])) / 2
        else:  # child x축이 parent y축과 정렬
            x_extent_c = (np.max(child_bbox[:, 1]) - np.min(child_bbox[:, 1])) / 2
            y_extent_c = (np.max(child_bbox[:, 0]) - np.min(child_bbox[:, 0])) / 2

        # 7. parent bbox 범위 계산 및 축소
        x_min_p = np.min(parent_bbox[:, 0])
        x_max_p = np.max(parent_bbox[:, 0])
        y_min_p = np.min(parent_bbox[:, 1])
        y_max_p = np.max(parent_bbox[:, 1])

        x_min = x_min_p + x_extent_c
        x_max = x_max_p - x_extent_c
        y_min = y_min_p + y_extent_c
        y_max = y_max_p - y_extent_c

        # 8. 축소된 bbox 생성 (z는 그대로 사용)
        z_val = np.mean(parent_bbox[:, 2])
        parent_obj_above_bbox = np.array([
            [x_min, y_min, z_val],
            [x_max, y_min, z_val],
            [x_max, y_max, z_val],
            [x_min, y_max, z_val],
        ])

        grid_centers_local = split_bbox_into_grid(parent_obj_above_bbox, num_rows=3, num_cols=3)  # (N, 3)



        # object local → world로 변환

        obj_pos_th = th.tensor(parent_pos, dtype=th.float32)
        obj_quat_th = th.tensor(parent_quat, dtype=th.float32)
        obj_T_world = pose2mat((obj_pos_th, obj_quat_th))  # (4, 4)

        grid_centers_local_th = th.tensor(grid_centers_local, dtype=th.float32)  # (N, 3)
        ones = th.ones((grid_centers_local_th.shape[0], 1))
        grid_centers_local_homo = th.cat([grid_centers_local_th, ones], dim=1)  # (N, 4)

        grid_centers_world_th = (obj_T_world @ grid_centers_local_homo.T).T[:, :3]
        grid_centers_world = grid_centers_world_th.numpy()  # 최종 (N, 3)

        # 1. (4, 3) → (4, 4) homogeneous 확장
        parent_bbox_local = th.tensor(parent_obj_above_bbox, dtype=th.float32)  # (4, 3)
        parent_bbox_local_homo = th.cat([parent_bbox_local, th.ones((4, 1))], dim=1)  # (4, 4)

        # 2. 로컬 → 월드 좌표 변환
        parent_bbox_world = (obj_T_world @ parent_bbox_local_homo.T).T[:, :3]  # (4, 3)

        # 3. numpy로 변환
        parent_obj_above_bbox_world = parent_bbox_world.numpy()

        # 3. 새 카메라 생성
        cam_name = f"vlm_camera_{parent_obj_name}"
        cam = VisionSensor(
            name=cam_name,
            relative_prim_path=f"/{cam_name}",
            # modalities=["rgb"],
            modalities=["rgb","seg_semantic", "seg_instance"],
            image_width=512,
            image_height=512,
            focal_length = 20.0
        )
        # scene.load(cam)
        cam.load(scene)

        # ========== Camera Parameter를 활용한 높이 설정 ==========

        # margin (안전 여유 계수)
        margin_scale = 1.1

        # 카메라 FOV 계산 (단위: 라디안)
        horizontal_fov = 2 * math.atan(cam.horizontal_aperture / (2 * cam.focal_length))
        vertical_fov = horizontal_fov * cam.image_height / cam.image_width

        # BBox의 x, y 크기 계산
        bbox_xs = parent_bbox[:, 0]
        bbox_ys = parent_bbox[:, 1]
        bbox_width = bbox_xs.max() - bbox_xs.min()
        bbox_height = bbox_ys.max() - bbox_ys.min()

        # 각각의 축 기준으로 필요한 카메라 높이 계산
        h_for_width = (bbox_width * margin_scale) / (2 * math.tan(horizontal_fov / 2))
        h_for_height = (bbox_height * margin_scale) / (2 * math.tan(vertical_fov / 2))

        # 둘 중 더 큰 값이 실제로 필요한 카메라 높이
        min_camera_height = max(h_for_width, h_for_height)


        # 카메라 위치 설정 (Z축 위쪽에 배치)
        cam_pos = parent_pos + np.array([0, 0, min_camera_height])

        look_dir = parent_rot_mat[:, 2]  # parent의 -Z 방향
        up_dir = parent_rot_mat[:, 1]    # parent의 +Y 방향 (혹은 world Y)

        # Look-at 행렬 생성
        camera_rot_mat = look_at_rotation(look_dir, up_dir)  # 이걸로 정확한 회전 생성


        # 회전 행렬 → 쿼터니언
        camera_quat = T.mat2quat(camera_rot_mat)

        # 카메라 pose 설정
        cam.set_position_orientation(cam_pos, camera_quat)
        # cam.set_position_orientation(cam_pos, [0, 0, 0, 1])  # 바라보는 방향은 그대로 (위에서 아래로)
        



        # # 위에서 바라보는 위치
        cam.initialize()
        # 1. 시뮬레이터 몇 프레임 실행 (렌더링 안정화용)
        for _ in range(5):
            og.sim.step()
            og.sim.render()

        # 2. Replicator 트리거: 데이터 캡처 요청
        rep.trigger.on_frame()

        # 3. 시뮬레이터 한 프레임 진행 (트리거 반영되도록)
        og.sim.step()
        og.sim.render()

        # 4. Replicator 데이터 수집 처리
        rep.orchestrator.step()


        pixel_coords = world_to_image(cam, grid_centers_world)

        # 5. 추가적으로 여유 프레임 (선택적)
        for _ in range(2):
            og.sim.step()
            og.sim.render()

        # # 0번, 1번 column 합쳐서 절대값 기준 max값
        # length_in_meters = np.max(np.abs(child_bbox[:, :2]))
        # print(length_in_meters)
        # distance_from_camera = cam_pos[2] - (parent_pos[2]+np.max(np.abs(child_bbox[:, 2])))
        # pixels = meters_to_pixels(length_in_meters, cam, distance_from_camera)
        # print(f"pixels: {pixels}")
        # # exit()
        
        obs, obs_info = cam.get_obs()

        rgb = obs["rgb"].cpu().numpy()[..., :3]
        seg = obs["seg_semantic"]
        label_map = obs_info["seg_semantic"]  # e.g., {825831922: 'keyboard', ...}
        seg_map_object = obs_info["seg_instance"]  # e.g., {825831922: 'keyboard', ...}


        object_items = list(seg_map_object.values())
        print(f"object_items: {object_items}")
        object_items.remove('groundPlane')
        object_items.remove(child_obj_name)
        # object_items.remove(parent_obj_name)
        
        print(f"object_items: {object_items}")

        all_object_info = {}
        for obj_name in object_items:
            obj = scene.object_registry("name", obj_name)
            
            # 위치 정보
            pos, quat = obj.get_position_orientation()
            pos = pos.cpu().numpy() if isinstance(pos, th.Tensor) else pos
            quat = quat.cpu().numpy() if isinstance(quat, th.Tensor) else quat

            # z축 위치가 parent보다 낮으면 건너뜀
            if pos[2] < parent_pos[2]:
                continue

            # BBox 정보
            bbox_dict = compute_obj_bbox_info(obj)
            bbox_top = bbox_dict["bbox_top_in_desired_frame"]

            bbox_world = transform_bbox_to_world(bbox_top, pos, quat)

            # 딕셔너리에 저장
            all_object_info[obj_name] = {
                "position": pos,
                "quaternion": quat,
                "bbox": bbox_top,
                "bbox_world": bbox_world
            }
        
        
        all_object_info["child_obj"] = {
                "position": child_pos,
                "quaternion": child_quat,
                "bbox": child_bbox,
                "bbox_world": transform_bbox_to_world(child_bbox, child_pos, child_quat)
            }

        # all_object_info["child_obj"] = all_object_info.pop(child_obj_name)
        all_object_info["parent_obj"] = all_object_info.pop(parent_obj_name)
        
        # # parent object ID 찾기
        # parent_id = [k for k, v in label_map.items() if v == parent_obj.category][0]
        
        # # binary mask 만들기 (parent object는 1, 나머지는 0)
        # binary_mask = (seg == parent_id).to(torch.uint8).cpu().numpy()

        # # 저장 (흑백 이미지로)
        # Image.fromarray(binary_mask * 255).save(f"{output_dir}/binary_mask.png")
        
        # exit()

        # 세그멘테이션 시각화 (라벨 텍스트 없이)
        seg_img_color = render_segmentation_overlay(seg, label_map)

        # ✅ 1. 불투명한 segmentation 저장 (alpha = 1.0)
        seg_img_opaque = np.concatenate([
            seg_img_color,
            255 * np.ones((seg_img_color.shape[0], seg_img_color.shape[1], 1), dtype=np.uint8)
        ], axis=-1)  # (H, W, 4)

        seg_img_opaque_pil = Image.fromarray(seg_img_opaque, mode="RGBA")
        seg_img_opaque_pil.save(os.path.join(output_dir, f"{file_prefix}_seg_semantic.png"))

        # ✅ 2. 반투명 버전으로 blending용 alpha 설정
        alpha_value = 0.2
        seg_img_color = np.concatenate([
            seg_img_color,
            int(alpha_value * 255) * np.ones((seg_img_color.shape[0], seg_img_color.shape[1], 1), dtype=np.uint8)
        ], axis=-1)  # (H, W, 4)

        # RGB → RGBA
        rgb = np.concatenate([
            rgb,
            255 * np.ones((rgb.shape[0], rgb.shape[1], 1), dtype=np.uint8)
        ], axis=-1)

        # NumPy → PIL 변환
        rgb_img = Image.fromarray(rgb, mode="RGBA")
        seg_img = Image.fromarray(seg_img_color, mode="RGBA")

        # 알파 블렌딩
        blended_img = Image.alpha_composite(rgb_img, seg_img)

        # 저장
        blended_img.save(os.path.join(output_dir, f"{file_prefix}_blended_overlay.png"))
        rgb_img.save(os.path.join(output_dir, f"{file_prefix}_view.png"))

        blended_number_img_path = os.path.join(output_dir, f"{file_prefix}_blended_overlay_with_numbers.png")
        blended_with_numbers = draw_numbers_on_blended_image(blended_img, pixel_coords)
        blended_with_numbers.save(blended_number_img_path)

        return blended_number_img_path, grid_centers_world, parent_obj_above_bbox_world, all_object_info

    @staticmethod
    def place_relative_to(
        count,
        scene,
        child_obj_name: str,
        parent_obj_name: str,
        placement: str = "above",
        parent_re_axis_mat: np.ndarray = np.diag([1, 1, 1]),
        child_re_axis_mat: np.ndarray = np.diag([1, 1, 1]),                                                                         
        above_sampled_point_3d=None,        
    ): 
        """
        Places the child link relative to the parent link's bounding box with a specified placement and clearance,
        accounting for the relative orientation of the links.
        """
        if not isinstance(parent_re_axis_mat, np.ndarray):
            parent_re_axis_mat = np.array(parent_re_axis_mat)
        if not isinstance(child_re_axis_mat, np.ndarray):
            child_re_axis_mat = np.array(child_re_axis_mat)

        parent_obj = scene.object_registry("name", parent_obj_name)
        parent_obj_bbox = compute_obj_bbox_info(parent_obj)
        
        parent_pos, parent_quat = parent_obj.get_position_orientation()
        parent_rot_mat = T.quat2mat(parent_quat.cpu().numpy())
        Translation = parent_pos.cpu().numpy()

        child_obj = scene.object_registry("name", child_obj_name)
        child_pos, _ = child_obj.get_position_orientation()

        # 기존 쿼터니언 → 회전 행렬
        child_rot_mat = T.quat2mat(parent_quat.cpu().numpy())

        # delta_rot = parent_re_axis_mat @ np.linalg.inv(child_re_axis_mat)
        delta_rot = parent_re_axis_mat @ np.linalg.inv(child_re_axis_mat)

        # delta 회전을 child에 적용
        final_rot_mat = delta_rot @ child_rot_mat

        # 최종 회전 행렬 → 쿼터니언
        final_child_quat = T.mat2quat(final_rot_mat)

        child_obj.set_position_orientation(
            th.tensor(child_pos, dtype=th.float),
            th.tensor(final_child_quat, dtype=th.float)
        )

        og.sim.step()


        child_pos, child_quat = child_obj.get_position_orientation()

        # child bbox 업데이트
        child_obj_bbox = compute_obj_bbox_info(child_obj)

        # ⬇️ 기본 방향 정의
        placement_directions = {
            "front":  np.array([1, 0, 0]),
            "back":   np.array([-1, 0, 0]),
            "left":   np.array([0, -1, 0]),
            "right":  np.array([0, 1, 0]),
            "above":  np.array([0, 0, 1]),
            "below":  np.array([0, 0, -1]),
        }

        local_offset = np.zeros(3)
        clearance = np.zeros(3)
        # if count==1:
        #     clearance = np.array([0.2, 0.3, 0])
        # else : 
        #     clearance = np.array([0.0, 0.6, 0])

        # child bbox (local 기준) → 보정된 child 기준으로 변환
        bbox_child_orig_local = child_obj_bbox["bbox_bottom_in_desired_frame"]
        # bbox_child_corrected_local = ((child_re_axis_mat) @ bbox_child_orig_local.T).T
        bbox_child_corrected_local = (np.linalg.inv(child_re_axis_mat) @ bbox_child_orig_local.T).T

        # bbox_child_corrected_local = child_obj_bbox["bbox_bottom_in_desired_frame"]
        
        
        
        clearance_axis = 0
        clearance_sign = 0
        if placement == "inside":
            local_offset = np.zeros(3)
        else:
            # child 좌표계 기준으로 parent obj가 어느 축에 있는지            

            placement_dir = parent_re_axis_mat @ placement_directions[placement]
            placement_axis = np.argmax(np.abs(placement_dir))
            placement_sign = int(np.sign(placement_dir[placement_axis]))


            # print("🧭 Placement 방향 계산")
            # print(f"  ▶ placement_axis: {placement_axis} ")
            # print(f"  ▶ placement_sign: {placement_sign} ")


            # ✅ 3. offset 계산 (parent 기준에서 위치 잡기)
            if placement_axis == 0:
                offset_val = np.max(parent_obj_bbox["bbox_bottom_in_desired_frame"][:, 0]) if placement_sign > 0 else np.min(parent_obj_bbox["bbox_bottom_in_desired_frame"][:, 0])
                base_offset = np.array([offset_val, 0.0, 0.0])
            elif placement_axis == 1:
                offset_val = np.max(parent_obj_bbox["bbox_bottom_in_desired_frame"][:, 1]) if placement_sign > 0 else np.min(parent_obj_bbox["bbox_bottom_in_desired_frame"][:, 1])
                base_offset = np.array([0.0, offset_val, 0.0])
            elif placement_axis == 2:
                offset_val = np.max(parent_obj_bbox["bbox_top_in_desired_frame"][:, 2]) if placement_sign > 0 else np.min(parent_obj_bbox["bbox_bottom_in_desired_frame"][:, 2])

                base_offset = np.array([0.0, 0.0, offset_val])
                # base_offset = np.array([x_max, y_max, offset_val])

            child_target_pos = parent_rot_mat @ base_offset + Translation

            

            child_obj.set_position_orientation(th.tensor(child_target_pos, dtype=th.float), th.tensor(child_quat, dtype=th.float))
            child_pos, child_quat = child_obj.get_position_orientation()

            # 기존 쿼터니언 → 회전 행렬
            child_rot_mat = T.quat2mat(child_quat.cpu().numpy())


            # parent_dir_local = np.linalg.inv(child_rot_mat @ child_re_axis_mat) @ (parent_pos - child_pos).cpu().numpy()
            parent_dir_local = np.linalg.inv(child_rot_mat) @ (parent_pos - child_pos).cpu().numpy()
            # print(child_obj_name)
            # print("parent_dir_local: ", parent_dir_local)
            # parent_dir_local = np.linalg.inv(parent_re_axis_mat) @ parent_dir_local

            # parent_dir_corrected_local = child_re_axis_mat @ parent_dir_local
            # print("parent_dir_corrected_local: ", parent_dir_corrected_local)
            clearance_axis = np.argmax(np.abs(parent_dir_local))
            clearance_sign = int(np.sign(parent_dir_local[clearance_axis]))
            # clearance_axis = np.argmax(np.abs(parent_dir_corrected_local))
            # clearance_sign = int(np.sign(parent_dir_corrected_local[clearance_axis]))

            # # 4. clearance axis & sign 계산
            # clearance_axis = np.argmax(np.abs(child_dir))
            # clearance_sign = int(np.sign(child_dir[clearance_axis]))


            print("\n📐 Clearance 방향 계산 (child local 기준)")
            print(f"  ▶ clearance_axis: {clearance_axis} ")
            print(f"  ▶ clearance_sign: {clearance_sign} ")


            # ✅ 4. clearance 계산 (child 보정 기준 bbox 사용)
            if clearance_axis == 0:
                clearance_val = -np.max(bbox_child_corrected_local[:, 0]) if clearance_sign > 0 else -np.min(bbox_child_corrected_local[:, 0])
            elif clearance_axis == 1:
                clearance_val = -np.max(bbox_child_corrected_local[:, 1]) if clearance_sign > 0 else -np.min(bbox_child_corrected_local[:, 1])
            elif clearance_axis == 2:
                clearance_val = -np.max(bbox_child_corrected_local[:, 2]) if clearance_sign > 0 else -np.min(bbox_child_corrected_local[:, 2])

            if placement_sign > 0:
                clearance_val = np.abs(clearance_val)
            else:
                clearance_val = -np.abs(clearance_val)


            placement_axis = np.argmax(np.abs(base_offset))
            
            clearance[placement_axis] = clearance_val
            print("clearance: ",clearance) # clearance_val:  0.013136092573404312

            local_offset = base_offset+clearance
            
            # ✅ 최종 위치 계산
            # child_target_pos = parent_rot_mat @ local_offset + Translation

            # child_obj.set_position_orientation(th.tensor(child_target_pos, dtype=th.float), th.tensor(child_quat, dtype=th.float))

            # for _ in range(10000):
            #     og.sim.render()

        # ✅ 최종 위치 계산
        child_target_pos = parent_rot_mat @ local_offset + Translation
        if placement == "above":
            child_target_pos[0] = above_sampled_point_3d[0]
            child_target_pos[1] = above_sampled_point_3d[1]
        return child_target_pos, child_quat, clearance_axis, clearance_sign

       
    def snap_to_place(
            scene,
            obj_info,
            cam_pose, 
            child_obj_name: str,
            parent_obj_name: str,
            placement: str = "above",
            
        ):
            """
            Snap the child object into contact with the parent object in the Z-axis direction, 
            either from above or below, using collision checking.
            """
            child_obj = scene.object_registry("name", child_obj_name)
            parent_obj = scene.object_registry("name", parent_obj_name)

            # 충돌 감지를 위해 두 객체 모두 물리 시뮬레이션 대상 + 시각화 객체로 설정

            old_state = og.sim.dump_state()

            step_size = 0.002  # 5mm 단위로 이동
            # print(type(obj_info["mount"]["wall"]))
            local_step_dir = np.zeros(3)            
            if placement == "below":
                local_step_dir = np.array([0, 0, 1.0]) 
            # elif (placement == "right" or placement=="left")and not obj_info["mount"]["floor"]:
            elif (placement == "right" or placement=="left"):
                # local_step_dir = np.array([0.0, -1.0, 0]) 
                # 이동해야 할 축 방향의 반대 방향으로 이동해서 충돌 확인
                local_step_dir[obj_info["clearance_axis"]] = obj_info["clearance_sign"]
            else:
                local_step_dir = np.array([0, 0, -1.0])


            # 현재 child의 중심 위치
            child_pos, child_quat = child_obj.get_position_orientation()

            rotation_matrix = T.quat2mat(child_quat.cpu().numpy())
            step_dir = rotation_matrix @ local_step_dir

            child_obj.keep_still()
            parent_obj.keep_still()
            child_obj.visual_only = False
            parent_obj.visual_only = False
            og.sim.step_physics()

            # Step 1: 만약 이미 충돌하고 있다면 → 반대 방향으로 빠질 때까지 이동
            if child_obj.states[Touching].get_value(parent_obj):
                if verbose := (child_obj_name == "task_obj_cup"):
                    print("🔄 Already touching, moving opposite direction to find separation")

                reverse_dir = -step_dir
                while child_obj.states[Touching].get_value(parent_obj):
                    og.sim.load_state(old_state)
                    new_pos = child_pos + th.tensor(reverse_dir, dtype=th.float) * step_size
                    if verbose:
                        print("  ⬅ Moving away from collision: ", new_pos)
                    child_obj.set_position_orientation(position=new_pos)
                    child_pos = new_pos
                    old_state = og.sim.dump_state()
                    og.sim.step_physics()
                    og.sim.step()
                    og.sim.render()

            # Step 2: 이제 충돌이 없으니 → 다시 원래 방향으로 붙을 때까지 이동
            if not child_obj.states[Touching].get_value(parent_obj):
                if verbose := (child_obj_name == "task_obj_cup"):
                    print("🔁 Moving to make contact")

                while not child_obj.states[Touching].get_value(parent_obj):
                    og.sim.load_state(old_state)
                    new_pos = child_pos + th.tensor(step_dir, dtype=th.float) * step_size
                    if verbose:
                        print("  ➡ Moving toward contact: ", new_pos)
                    child_obj.set_position_orientation(position=new_pos)
                    child_pos = new_pos
                    old_state = og.sim.dump_state()
                    og.sim.step_physics()
                    og.sim.step()
                    og.sim.render()

                # 충돌 직전 상태로 복원 (1 step back)
                og.sim.load_state(old_state)
                final_pos = child_pos - th.tensor(step_dir, dtype=th.float) * step_size
                child_obj.set_position_orientation(position=final_pos)

                # 최종 pose를 tf_from_cam으로 업데이트
                obj_pos, obj_quat = child_obj.get_position_orientation()
                rel_tf = T.relative_pose_transform(
                    obj_pos.cpu().detach().numpy(),
                    obj_quat.cpu().detach().numpy(),
                    cam_pose[0],
                    cam_pose[1],
                )
                obj_info["tf_from_cam"] = T.pose2mat(rel_tf)

            # # 만약 현재 충돌하고 있지 않다면 → 충돌이 발생할 때까지 아래로(또는 위로) 이동
            # if not child_obj.states[Touching].get_value(parent_obj):
            #     while not child_obj.states[Touching].get_value(parent_obj):
            #         if child_obj_name =="task_obj_cup":
            #             print(not child_obj.states[Touching].get_value(parent_obj))
                    
            #         og.sim.load_state(old_state)
            #         new_pos = child_pos + th.tensor(step_dir, dtype=th.float) * step_size
            #         if child_obj_name =="task_obj_cup":
            #             print("new pose : ", new_pos)
            #         child_obj.set_position_orientation(position=new_pos)
            #         child_pos = new_pos
            #         old_state = og.sim.dump_state()
            #         og.sim.step_physics()
            #         og.sim.step()
            #         og.sim.render()

            #     # 충돌 직전 상태로 복원 (1 step back)
            #     og.sim.load_state(old_state)
            #     final_pos = child_pos - th.tensor(step_dir, dtype=th.float) * step_size
            #     child_obj.set_position_orientation(position=final_pos)

            #     # 최종 pose를 tf_from_cam으로 업데이트
            #     obj_pos, obj_quat = child_obj.get_position_orientation()
            #     rel_tf = T.relative_pose_transform(
            #         obj_pos.cpu().detach().numpy(),
            #         obj_quat.cpu().detach().numpy(),
            #         cam_pose[0],
            #         cam_pose[1],
            #     )
            #     obj_info["tf_from_cam"] = T.pose2mat(rel_tf)

            # else:
            #     og.sim.load_state(old_state)

            # # 다시 visual only로 설정
            # child_obj.visual_only = True
            # parent_obj.visual_only = True
            # og.sim.step_physics()
            # og.sim.step()



    # def random_position_on_parent(            
    #         scene,
    #         obj_info,
    #         cam_pose, 
    #         child_obj_name: str,
    #         parent_obj_name: str,
    #         parent_re_axis_mat: np.ndarray = np.diag([1, 1, 1]),
    #         child_re_axis_mat: np.ndarray = np.diag([1, 1, 1]),
    #         ):    
    #     """
    #     Generates a random position on the parent bounding box for placing the child object.

    #     Args:
    #         parent_bbox (np.ndarray): The bounding box of the parent object.
    #         child_bbox_local_corrected (np.ndarray): The corrected bounding box of the child object.

    #     Returns:
    #         tuple: Random x and y coordinates for placing the child object.
    #     """
    #     if not isinstance(parent_re_axis_mat, np.ndarray):
    #         parent_re_axis_mat = np.array(parent_re_axis_mat)
    #     if not isinstance(child_re_axis_mat, np.ndarray):
    #         child_re_axis_mat = np.array(child_re_axis_mat)


    #     parent_obj = scene.object_registry("name", parent_obj_name)
    #     parent_obj_bbox = compute_obj_bbox_info(parent_obj)

    #     parent_pos, parent_quat = parent_obj.get_position_orientation()

    #     parent_rot_mat = T.quat2mat(parent_quat.cpu().numpy())



    #     child_obj = scene.object_registry("name", child_obj_name)
    #     child_pos, child_quat = child_obj.get_position_orientation()

    #     Translation = child_pos.cpu().numpy()

    #     child_obj_bbox_raw = compute_obj_bbox_info(child_obj)

 

    #     child_bbox_raw = child_obj_bbox_raw["bbox_bottom_in_desired_frame"]
    #     parent_bbox = parent_obj_bbox["bbox_bottom_in_desired_frame"]
    #     child_bbox = (np.linalg.inv(child_re_axis_mat) @ child_bbox_raw.T).T

    #     x_min_p = np.min(parent_bbox[:, 0])
    #     x_max_p = np.max(parent_bbox[:, 0])
    #     y_min_p = np.min(parent_bbox[:, 1])
    #     y_max_p = np.max(parent_bbox[:, 1])

    #     R_rel = parent_re_axis_mat.T @ child_re_axis_mat
    #     vec = R_rel @ np.array([1, 0, 0])  # child x축 in parent frame
    #     child_axis = np.argmax(np.abs(vec))
    #     if not child_axis:
    #         x_extent_c = (np.max(child_bbox[:, 0]) - np.min(child_bbox[:, 0])) / 2
    #         y_extent_c = (np.max(child_bbox[:, 1]) - np.min(child_bbox[:, 1])) / 2    
    #     else:
    #         x_extent_c = (np.max(child_bbox[:, 1]) - np.min(child_bbox[:, 1])) / 2
    #         y_extent_c = (np.max(child_bbox[:, 0]) - np.min(child_bbox[:, 0])) / 2

    #     x_min = x_min_p + x_extent_c
    #     x_max = x_max_p - x_extent_c
    #     y_min = y_min_p + y_extent_c
    #     y_max = y_max_p - y_extent_c

    #     random_x = random.uniform(x_min, x_max)
    #     random_y = random.uniform(y_min, y_max)


    #     # base_offset = [x_max, y_max, 0.0]
    #     base_offset = [0.0, 0.0, 0.0]


            
    #         # ✅ 최종 위치 계산
    #         # child_target_pos = parent_rot_mat @ local_offset + Translation

    #         # child_obj.set_position_orientation(th.tensor(child_target_pos, dtype=th.float), th.tensor(child_quat, dtype=th.float))

    #         # for _ in range(10000):
    #         #     og.sim.render()

    #     # ✅ 최종 위치 계산
    #     child_target_pos = parent_rot_mat @ base_offset + Translation
        
    #     child_obj.set_position_orientation(th.tensor(child_target_pos, dtype=th.float), th.tensor(child_quat, dtype=th.float))


    #     child_obj.keep_still()
    #     parent_obj.keep_still()
    #             # ✅ 모든 오브젝트의 물리 충돌 허용
    #     for obj in scene.objects:
    #         obj.visual_only = False  # 충돌체 활성화
    #     # child_obj.visual_only = False
    #     # parent_obj.visual_only = False
    #     og.sim.step_physics()
    #     og.sim.render()

        
    #     # # ✅ 충돌 검사 (모든 오브젝트 대상)
    #     # og.sim.step_physics()
    #     # og.sim.step()

    #     is_touching_any = False
    #     for obj in scene.objects:
    #         print(f"obj name: {obj.name}")
    #         if obj.name == child_obj.name:
    #             continue  # 자기 자신은 제외
    #         if child_obj.states[Touching].get_value(obj):
    #             print(f"[⚠️] Collision detected between '{child_obj_name}' and '{obj.name}'")
    #             is_touching_any = True

    #     print(f"[INFO] Collision with any object? -> {is_touching_any}")
    #     return is_touching_any

    def above_collision_check(
            scene,
            obj_info,
            child_obj_name,
            parent_obj_name,
            above_sampled_point_3d=None,
            probability_map_sampler=None,
        ):

        assert above_sampled_point_3d is not None, "above_sampled_point_3d must be provided."

        def check_valid_pose(child_obj, dist_diff=2.0, x_diff=1.0, y_diff=1.0):
            """
            Returns True if the object is NOT in collision (i.e., placement is valid).
            """
            for all_obj in scene.objects:
                all_obj.visual_only = False
                all_obj.keep_still()

            og.sim.step_physics()
            for _ in range(20):
                og.sim.step()
                og.sim.render()
            new_child_pos, new_child_quat = child_obj.get_position_orientation()
            
            dist = np.linalg.norm(np.array(above_sampled_point_3d) - np.array(new_child_pos))
            

            x_diff = abs(above_sampled_point_3d[0] - new_child_pos[0])
            y_diff = abs(above_sampled_point_3d[1] - new_child_pos[1])

            for all_obj in scene.objects:
                all_obj.visual_only = True

            if dist < 0.35 and x_diff < 0.01 and y_diff < 0.01:
                print("Successfully placed the object!")
                return True
            else:
                print("dist: ", dist)
                print("x_diff: ", x_diff)
                print("y_diff: ", y_diff)
                print("Failed to place the object, trying again...")
            
            # print("###################################################################################")

            # in_collision = check_collision(prims=obj, step_physics=False)
            # contacts = obj.contact_list()

            
            return False

        child_obj = scene.object_registry("name", child_obj_name)
        parent_obj = scene.object_registry("name", parent_obj_name)
        
        child_pos, child_quat = child_obj.get_position_orientation()
        parent_pos, parent_quat = parent_obj.get_position_orientation()

        child_pos = child_pos.cpu().numpy().copy()  # numpy로 복사
        child_quat = child_quat.cpu().numpy().copy()

        
        while True:
            # print("above_sampled_point_3d: ", above_sampled_point_3d)
            above_sampled_point_3d = probability_map_sampler.sample()
            # print(child_pos)

            child_pos[0] = above_sampled_point_3d[0]
            child_pos[1] = above_sampled_point_3d[1]
            child_obj.set_position_orientation(
                        th.tensor(child_pos, dtype=th.float),
                        th.tensor(child_quat, dtype=th.float),
                    )
            
            original_state = og.sim.dump_state()

            if check_valid_pose(child_obj):
                # og.sim.load_state(original_state)
                child_obj.set_position_orientation(
                        th.tensor(child_pos, dtype=th.float),
                        th.tensor(child_quat, dtype=th.float),
                    )
                break
        
        # return False

    def enable_collision_and_physics(scene, child_obj_name, parent_obj_name, step_frames=10):
        """
        Enables collision and physics (gravity, dynamics) for parent and child objects.
        """
        child_obj = scene.object_registry("name", child_obj_name)
        parent_obj = scene.object_registry("name", parent_obj_name)

        # visual_only=False => 시뮬레이션에 포함, collision 및 중력 적용됨
        child_obj.visual_only = False
        parent_obj.visual_only = False

        # 물리 엔진과 상호작용 위해 정지 상태 설정 (optional)
        child_obj.keep_still()
        parent_obj.keep_still()

        og.sim.step_physics()
        for _ in range(step_frames):
            og.sim.step()
            og.sim.render()

        print(f"Collision and physics enabled for: {child_obj_name}, {parent_obj_name}")



    def random_position_on_parent(
        scene,
        obj_info,
        cam_pose,
        child_obj_name: str,
        parent_obj_name: str,
        parent_re_axis_mat: np.ndarray = np.diag([1, 1, 1]),
        child_re_axis_mat: np.ndarray = np.diag([1, 1, 1]),
    ):
        """
        Tries random placements of child_obj on parent_obj until there is no collision.
        """
        if not isinstance(parent_re_axis_mat, np.ndarray):
            parent_re_axis_mat = np.array(parent_re_axis_mat)
        if not isinstance(child_re_axis_mat, np.ndarray):
            child_re_axis_mat = np.array(child_re_axis_mat)

        def check_valid_pose(obj, pos, quat):
            """
            Returns True if child_obj is touching any other object in the scene.
            """

            # Make sure sim is playing
            assert og.sim.is_playing(), "Cannot test valid pose while sim is not playing!"
            

            # Store state before checking object position
            state = og.sim.dump_state()

            for all_obj in scene.objects:
                all_obj.visual_only = False



            # Set the pose of the object
            print(pos)
            # place_base_pose(obj, pos, quat)
            obj.set_position_orientation(
                    th.tensor(pos, dtype=th.float),
                    th.tensor(quat, dtype=th.float),
                )

            og.sim.step_physics()
            contacts = obj.contact_list()
            if contacts:
                print(f"🔧 Contacts for {obj.name}:")
                for c in contacts:
                    print(f"  - {c.body0} <--> {c.body1}")
            for _ in range(1000):
                og.sim.render()


            # obj.keep_still()

            # Check whether we're in collision after taking a single physics step
            in_collision = check_collision(prims=obj, step_physics=False)
            contacts = obj.contact_list()
            print("Not Visible Only")
            if contacts:
                print(f"🔧 Contacts for {obj.name}:")
                for c in contacts:
                    print(f"  - {c.body0} <--> {c.body1}")

            # Restore state after checking the collision
            og.sim.load_state(state)

            for all_obj in scene.objects:
                all_obj.visual_only = True

            # Valid if there are no collisions
            return not in_collision

        parent_obj = scene.object_registry("name", parent_obj_name)
        parent_obj_bbox = compute_obj_bbox_info(parent_obj)
        parent_pos, parent_quat = parent_obj.get_position_orientation()
        parent_rot_mat = T.quat2mat(parent_quat.cpu().numpy())

        child_obj = scene.object_registry("name", child_obj_name)
        child_pos, child_quat = child_obj.get_position_orientation()
        print("child_quat:",child_quat)
        Translation = child_pos.cpu().numpy()
        child_obj_bbox_raw = compute_obj_bbox_info(child_obj)
        child_bbox_raw = child_obj_bbox_raw["bbox_bottom_in_desired_frame"]
        parent_bbox = parent_obj_bbox["bbox_bottom_in_desired_frame"]
        child_bbox = (np.linalg.inv(child_re_axis_mat) @ child_bbox_raw.T).T

        x_min_p = np.min(parent_bbox[:, 0])
        x_max_p = np.max(parent_bbox[:, 0])
        y_min_p = np.min(parent_bbox[:, 1])
        y_max_p = np.max(parent_bbox[:, 1])

        R_rel = parent_re_axis_mat.T @ child_re_axis_mat
        vec = R_rel @ np.array([1, 0, 0])
        child_axis = np.argmax(np.abs(vec))
        if not child_axis:
            x_extent_c = (np.max(child_bbox[:, 0]) - np.min(child_bbox[:, 0])) / 2
            y_extent_c = (np.max(child_bbox[:, 1]) - np.min(child_bbox[:, 1])) / 2
        else:
            x_extent_c = (np.max(child_bbox[:, 1]) - np.min(child_bbox[:, 1])) / 2
            y_extent_c = (np.max(child_bbox[:, 0]) - np.min(child_bbox[:, 0])) / 2

        x_min = x_min_p + x_extent_c
        x_max = x_max_p - x_extent_c
        y_min = y_min_p + y_extent_c
        y_max = y_max_p - y_extent_c



        # 반복 샘플링
        max_tries = 20
        # random.seed(42)
        for i in range(max_tries):
            random_x = random.uniform(x_min, x_max)
            random_y = random.uniform(y_min, y_max)
            base_offset = [random_x, random_y, 0.0]
            # base_offset = [0, 0, 0.0]

            child_target_pos = parent_rot_mat @ base_offset + Translation
            print("child_target_pos:", child_target_pos)
            print("child_quat:", child_quat)
            print(f"[{i}] Trying placement at (x={random_x:.3f}, y={random_y:.3f})")
            state = og.sim.dump_state()
            is_valid = check_valid_pose(
                obj=child_obj,
                pos=th.tensor(child_target_pos, dtype=th.float),  # ⬅ torch 텐서로 변환
                quat=th.tensor(child_quat, dtype=th.float)
            )

            if is_valid:
                # 최종적으로 pose 적용
                child_obj.set_position_orientation(
                    th.tensor(child_target_pos, dtype=th.float),
                    th.tensor(child_quat, dtype=th.float),
                )
                print(f"[✅] Found collision-free position on try {i}")
                return True
            og.sim.load_state(state)

        print(f"[❌] Failed to find collision-free placement after {max_tries} tries")
        return False

        # return is_touching


    def align_object_z_axis(scene, obj_info, cam_pose, child_obj_name, verbose=True):
        """
        Aligns the object's z-axis with world +z using base-aligned bounding box (BAB).
        Also sets the correct position so the object stays in place.

        Args:
            scene: OG scene
            obj_info (dict): Metadata about the object (not used currently)
            cam_pose (tuple): (position, quaternion) of the camera (not used currently)
            child_obj_name (str): Object name in scene
            verbose (bool): If True, prints info

        Returns:
            None
        """
        # Get object from scene
        obj = scene.object_registry("name", child_obj_name)

        # Step 1: Reset pose to eliminate existing rotation/scale artifacts
        obj.set_position_orientation(
            position=th.tensor([0, 0, 0], dtype=th.float),
            orientation=th.tensor([0, 0, 0, 1], dtype=th.float)
        )
        obj.keep_still()
        og.sim.step_physics()

        # Step 2: Get base-aligned orientation and position (key step!)
        bbox_center_world, bbox_orn_in_world, _, _ = obj.get_base_aligned_bbox()
        bbox_quat = bbox_orn_in_world.cpu().detach().numpy()
        bbox_center = bbox_center_world.cpu().detach().numpy()

        # Step 3: Apply corrected position + orientation
        obj.set_position_orientation(
            position=th.tensor(bbox_center, dtype=th.float),
            orientation=th.tensor(bbox_quat, dtype=th.float),
        )
        og.sim.step_physics()

        if verbose:
            print(f"✅ '{child_obj_name}' 정렬 완료 (base-aligned z-up).")
            print(f"  - 위치: {bbox_center}")
            print(f"  - 쿼터니언: {bbox_quat}")


    def world_to_local(
    world_position: List[float],
    reference_orientation: List[float],
    reference_base_pos: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        Converts a target position from the world frame to a link's local frame.
        `reference_base_pos` is used e.g, when we want to compute the offset to the parent origin
        (i.e., the origin tag of the joint) such that we're shifted to `world_position`. The more common
        use case is when `world_position` is already just the offset (i.e., for `place_relative_to` function)

        Args:
            world_position (list): The target position in the world frame.
            reference_orientation (list): The orientation of the reference link.
            reference_base_pos (list, optional): The base position of the reference link in the world frame.

        Returns:
            numpy.ndarray: The position in the local frame of the reference link.
        """
        if reference_base_pos is not None:
            world_position = np.array(world_position) - \
                np.array(reference_base_pos)
        ref_rotation = R.from_quat(reference_orientation)
        local_position = ref_rotation.inv().apply(world_position)
        return local_position

class ProbMapSampler:
    def __init__(self, prob_map, bounds, origin, x_axis, y_axis):
        self.prob_map = prob_map
        self.bounds = bounds
        self.origin = origin
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.grid_res = prob_map.shape[0]

        # 미리 계산
        self.u_vals = np.linspace(bounds[0], bounds[1], self.grid_res)
        self.v_vals = np.linspace(bounds[2], bounds[3], self.grid_res)
        self.flat_probs = prob_map.flatten()
        self.flat_probs = self.flat_probs / np.sum(self.flat_probs)


    def sample(self):
        idx = np.random.choice(len(self.flat_probs), p=self.flat_probs)
        i, j = divmod(idx, self.grid_res)
        u = self.u_vals[j]
        v = self.v_vals[i]
        return self.origin + u * self.x_axis + v * self.y_axis  # (3,) np.ndarray
