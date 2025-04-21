from threading import local
from robomimic.utils.log_utils import PrintLogger
import torch as th
import numpy as np
from pathlib import Path
from PIL import Image
from copy import deepcopy
import os
import json
import imageio
from typing import Optional, List
import omnigibson as og
from omnigibson.scenes import Scene
from omnigibson.objects import DatasetObject
from omnigibson.object_states import Touching
from omnigibson.object_states import ToggledOn
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

        scene_rgb = self.take_photo(n_render_steps=10)

        # scene_rgb = self.joint_test(scene, n_render_steps=10)

        with open(task_feature_matching_path, "r") as f:
            task_obj_output_info = json.load(f)

        scene = TaskSceneGenerator.add_task_object(scene=scene, scene_info=scene_info, cam_pose=cam_pose, obj_info_json=task_obj_output_info, visual_only=True)
        # scene_rgb = self.take_photo(n_render_steps=1000000)
        scene_rgb = self.take_photo(n_render_steps=100)
        # exit(())
        
        final_scene_info = deepcopy(scene_info)

        # Save final info
        with open(f"{save_dir}/scene_info.json", "w+") as f:
            json.dump(final_scene_info, f, indent=4, cls=NumpyTorchEncoder)
        print(final_scene_info)
        exit()


        # Object BBox 정보 저장
        if self.verbose:
            print("Object BBox 정보 저장")
        all_obj_bbox_info = dict()
        for obj_name, obj_info in scene_info["objects"].items():
            if discard_objs and obj_name in discard_objs:
                continue
            if self.verbose:
                print(f"obj_name: {obj_name}")
                print(f"obj_info: {obj_info}")
            # Grab object and relevant info
            obj = scene.object_registry("name", obj_name)
            obj_bbox_info = compute_obj_bbox_info(obj=obj)
            if self.verbose:
                print(f"obj_bbox_info: {obj_bbox_info}")
            # obj_bbox_info["articulated"] = step_2_output_info["objects"][obj_name]["articulated"]
            obj_bbox_info["mount"] = obj_info["mount"]
            all_obj_bbox_info[obj_name] = obj_bbox_info
        sorted_z_obj_bbox_info = dict(sorted(all_obj_bbox_info.items(), key=lambda x: x[1]['lower'][2]))  # sort by lower corner's height (z)
        print("##############################################")
        print(sorted_z_obj_bbox_info)
        exit()

        scene_graph_info = {
            "floor": {
                "objOnTop": [],
                "objBeneath": None,  # This must be empty, i.e., no obj is beneath floor
                "mount": {
                    "floor": True,
                    "wall": False,
                },
            },
        }

        final_scene_info = deepcopy(scene_info)
        for name in sorted_z_obj_bbox_info:
            #### 높이 조정 ####
                obj_name_beneath, z_offset = compute_object_z_offset_non_articulated(
                    target_obj_name=name,
                    sorted_obj_bbox_info=sorted_z_obj_bbox_info,
                    verbose=self.verbose,
                )
                obj = scene.object_registry("name", name)

                #### 바닥에 꼭 있어야 하는 객체 처리 ####
                if scene_info["objects"][name]["category"] in CATEGORIES_MUST_ON_FLOOR:
                    obj_name_beneath = "floor"
                    z_offset = -sorted_z_obj_bbox_info[name]["lower"][-1]

                # Add information to scene graph info
                if name not in scene_graph_info.keys():
                    scene_graph_info[name] = {
                        "objOnTop": [],
                        "objBeneath": obj_name_beneath,
                        "mount": None,
                    }
                else:
                    scene_graph_info[name]["objBeneath"] = obj_name_beneath

                if obj_name_beneath not in scene_graph_info.keys():
                    scene_graph_info[obj_name_beneath] = {
                        "objOnTop": [name],
                        "objBeneath": None,
                        "mount": None,
                    }
                else:
                    scene_graph_info[obj_name_beneath]["objOnTop"].append(name)

                mount_type = scene_info["objects"][name]["mount"]  # a list
                scene_graph_info[name]["mount"] = mount_type
                #### 객체 고정 ####
                # TODO
                obj.keep_still()

                # Modify object pose if z_offset is not 0
                if z_offset != 0:
                    if (not mount_type["floor"]) and z_offset <= 0:
                        # If the object in mounted on the wall, and we want to lower it, omit that
                        continue
                    new_center = sorted_z_obj_bbox_info[name]["center"] + np.array([0.0, 0.0, z_offset])
                    ##### 객체 위치 재설정 ####
                    obj.set_bbox_center_position_orientation(position=th.tensor(new_center, dtype=th.float), orientation=None)
                    og.sim.step_physics()

                    #### 높이 조정 후 Json BBox 업데이트
                    # Grab updated obj bbox info
                    obj_bbox_info = compute_obj_bbox_info(obj=obj)
                    sorted_z_obj_bbox_info[name].update(obj_bbox_info)

                #### 카메라 기준 변환 행렬 업데이트 ####
                # Update scene_info
                obj_pos, obj_quat = obj.get_position_orientation()
                rel_tf = T.relative_pose_transform(obj_pos.cpu().detach().numpy(), obj_quat.cpu().detach().numpy(), cam_pose[0], cam_pose[1])
                final_scene_info["objects"][name]["tf_from_cam"] = T.pose2mat(rel_tf)

        for obj in scene.objects:
            obj.keep_still()
        og.sim.step_physics()

        for _ in range(3):
            og.sim.render()

        sorted_x_obj_bbox_info = dict(sorted(sorted_z_obj_bbox_info.items(), key=lambda x: x[1]['lower'][0], reverse=True))  # sort by lower corner's x
        obj_names = list(sorted_x_obj_bbox_info.keys())

        if self.verbose:
            print(f"[Scene] depenetrating collisions...")

        # Iterate over all objects; check for collision
        for obj1_idx, obj1_name in enumerate(obj_names):

            # 투명하거나 충돌이 필요없는 것들을 검사 제외
            # Skip any non-collidable categories
            if any(cat in obj1_name for cat in NON_COLLIDABLE_CATEGORIES):
                continue

            # Grab the object, make it collidable
            obj1 = scene.object_registry("name", obj1_name)
            # TODO
            obj1.keep_still()
            obj1.visual_only = False

            # 왼쪽에서 부터 점점 오른쪽하고 충돌 비교
            # Check all subsequent downstream objects for collision
            for obj2_name in obj_names[obj1_idx + 1:]:
                
                # Skip any non-collidable categories
                if any(cat in obj2_name for cat in NON_COLLIDABLE_CATEGORIES):
                    continue

                # Sanity check to make sure the two objects aren't the same
                assert obj1_name != obj2_name

                # 수직관계로 놓여있을 경우 충돌 제외
                # If the objects are related by a vertical relationship, continue -- collision is expected
                if (obj2_name in scene_graph_info[obj1_name]['objOnTop']) or (
                        scene_graph_info[obj1_name]["objBeneath"] == obj2_name):
                    continue
                
                ## 충돌 검사 ##
                # Grab the object, make it collidable
                obj2 = scene.object_registry("name", obj2_name)
                old_state = og.sim.dump_state()
                # TODO
                obj2.keep_still()
                obj2.visual_only = False
                og.sim.step_physics()

                obj12_collision = obj2.states[Touching].get_value(obj1)
                ##############

                # If we're in contact, move the object with smaller x value
                if obj12_collision:
                    # Adjust the object with smaller x
                    if self.verbose:
                        print(f"Detected collision between {obj1_name} and {obj2_name}")
                    # Get obj 2's x and y axes
                    obj2_ori_mat = T.quat2mat(obj2.get_position_orientation()[1].cpu().detach().numpy())
                    obj2_x_dir = obj2_ori_mat[:, 0]
                    obj2_y_dir = obj2_ori_mat[:, 1]

                    center_step_size = 0.01  # 1cm
                    obj2_to_obj1 = (obj1.get_position_orientation()[0] - obj2.get_position_orientation()[0]).cpu().detach().numpy()

                    chosen_axis = obj2_x_dir if abs(np.dot(obj2_x_dir, obj2_to_obj1)) > abs(np.dot(obj2_y_dir, obj2_to_obj1)) else obj2_y_dir
                    center_step_dir = -chosen_axis if np.dot(chosen_axis, obj2_to_obj1) > 0 else chosen_axis

                    # 충돌이 없을 때까지 이동
                    while obj2.states[Touching].get_value(obj1):
                        og.sim.load_state(old_state)
                        new_center = obj2.get_position_orientation()[0] + th.tensor(center_step_dir, dtype=th.float) * center_step_size
                        obj2.set_position_orientation(position=new_center)
                        old_state = og.sim.dump_state()
                        og.sim.step_physics()

                    # 충돌 해결후 새로운 위치 설정
                    # Finally, load the collision-free state, update relative transformation
                    og.sim.load_state(old_state)
                    obj2.set_position_orientation(position=new_center)
                    obj_pos, obj_quat = obj2.get_position_orientation()
                    rel_tf = T.relative_pose_transform(obj_pos.cpu().detach().numpy(), obj_quat.cpu().detach().numpy(), cam_pose[0], cam_pose[1])
                    final_scene_info["objects"][obj2_name]["tf_from_cam"] = T.pose2mat(rel_tf)
                else:
                    # Simply load old state
                    og.sim.load_state(old_state)
                # Make obj2 visual only again so as not collide with any other objects
                # obj2.visual_only = False
                obj2.visual_only = True
            # Make obj1 visual only again so as not collide with any other objects
            # obj1.visual_only = False
            obj1.visual_only = True
        
        for obj in scene.objects:
            obj.keep_still()
            og.sim.step_physics()
        
        # print(sorted_z_obj_bbox_info)
        scene_rgb = self.take_photo(n_render_steps=10000)
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
    
    def add_task_object(scene, scene_info, cam_pose, obj_info_json, visual_only=False):
         # Load all objects
        with og.sim.stopped():
            for obj_name, obj_info in obj_info_json["objects"].items():
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
                obj_pos, obj_quat, child_clearance_axis, child_clearance_sign = TaskSceneGenerator.place_relative_to(scene, 
                                                                         child_obj_name=child_obj_name,
                                                                         parent_obj_name=parent_obj_name,
                                                                         placement=placement,
                                                                         parent_re_axis_mat=obj_info.get("parent_re_axis_mat", np.diag([1, 1, 1])),
                                                                         child_re_axis_mat=obj_info.get("re_axis_mat", np.diag([1, 1, 1]))
                                                                        )
                obj_info_json["objects"][obj_name]["clearance_axis"] = child_clearance_axis
                obj_info_json["objects"][obj_name]["clearance_sign"] = child_clearance_sign

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
        for obj_name, obj_info in obj_info_json["objects"].items():
            print("obj_name: ", obj_name)
            TaskSceneGenerator.snap_to_place(scene, 
                                            obj_info,
                                            cam_pose=cam_pose,
                                            child_obj_name=obj_name,
                                            parent_obj_name=obj_info["parent_object"],
                                            placement=obj_info["placement"])
            
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
    def place_relative_to(
        scene, 
        child_obj_name: str,
        parent_obj_name: str,
        placement: str = "above",
        parent_re_axis_mat: np.ndarray = np.diag([1, 1, 1]),
        child_re_axis_mat: np.ndarray = np.diag([1, 1, 1]),
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

        # child bbox (local 기준) → 보정된 child 기준으로 변환
        bbox_child_orig_local = child_obj_bbox["bbox_bottom_in_desired_frame"]
        print(bbox_child_orig_local)
        # bbox_child_corrected_local = ((child_re_axis_mat) @ bbox_child_orig_local.T).T
        bbox_child_corrected_local = (np.linalg.inv(child_re_axis_mat) @ bbox_child_orig_local.T).T

        print(bbox_child_corrected_local)
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
        
            local_offset = base_offset+clearance
            
            # ✅ 최종 위치 계산
            # child_target_pos = parent_rot_mat @ local_offset + Translation

            # child_obj.set_position_orientation(th.tensor(child_target_pos, dtype=th.float), th.tensor(child_quat, dtype=th.float))

            # for _ in range(10000):
            #     og.sim.render()

        # ✅ 최종 위치 계산
        child_target_pos = parent_rot_mat @ local_offset + Translation

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

            step_size = 0.005  # 5mm 단위로 이동
            # print(type(obj_info["mount"]["wall"]))
            local_step_dir = np.zeros(3)            
            if placement == "below":
                local_step_dir = np.array([0, 0, 1.0]) 
            elif (placement == "right" or placement=="left")and not obj_info["mount"]["floor"]:
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
