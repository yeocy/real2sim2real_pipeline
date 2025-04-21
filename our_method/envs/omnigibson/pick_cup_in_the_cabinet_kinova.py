from copy import deepcopy
from collections.abc import Iterable
import random
import json


import torch as th
from our_method.envs.omnigibson.skill_wrapper import SkillWrapper
from our_method.skills.open_and_grab_kinova import OpenandGrabSkill
from our_method.skills.open_or_close_skill import OpenOrCloseSkill
import omnigibson as og
from omnigibson.objects import DatasetObject
from omnigibson.prims.material_prim import MaterialPrim
from omnigibson.utils.asset_utils import get_all_object_category_models
import omnigibson.utils.transform_utils as OT
from omnigibson.object_states import ToggledOn


class PickCupInTheCabinetKinovaWrapper(SkillWrapper):
    """
    An OmniGibson environment wrapper for instantiating a specific OpenCabinet task

    Args:
        env (Environment): The omnigibson environment to wrap
        eef_z_offset (float): Distance in the robot's EEF z-direction specifying distance to its actual grasping
            location for its assumed parallel jaw gripper
        cab_categories (str or list of str): Cabinet categor(ies) to use. Default is "bottom_cabinet"
        cab_models (None or str or list of str): Cabinet model(s) to use. If None, will sample randomly
        cab_links (None or str or list of str): Cabinet link name(s) to use. If None, will sample randomly
        cab_bboxs (None or 3-array or list of 3-array): If specified, (x,y,z) bounding box(s) for the cabinet(s).
            If None, will use the default bbox of the object
        eval_idx (None or int): If specified, the index in @cab_models array to use for evaluating this environment.
            If None, will randomly sample one at every episode reset
        handle_dist (float): Distance into a detected handle to move the robot EEF
        approach_dist (float): Distance when executing skill for approaching cabinet handle
        dist_out_from_handle (float): Distance outwards from handle to grab for placing robot base
        dist_right_of_handle (float): Distance rightwards from handle to grab for placing robot base
        dist_up_from_handle (float): Distance upwards from handle to grab for placing robot base
        z_rot_from_handle (float): Z-rotation default value with respect to the furniture object
        xyz_randomization (None or 3-array): If specified, max (x,y,z) randomization to apply to all objects
            (besides agent). If None, will use 0
        z_rot_randomization (None or float): If specified, max z-rotation randomization to apply to all objects
            (besides agent). If None, will use 0
        bbox_randomization (3-array or list of array): Randomization to apply per-bounding box -- each value should be
            a (dx,dy,dz) +/- value to apply randomization. If single value, will be broadcast across all models
        randomize_textures (bool or list of bool): Whether textures should be randomized or not per-model. If a single
            bool is given, will be broadcast across all models
        randomize_cabinet_pose (bool): Whether to randomize the cabinet pose or not,
        randomize_agent_pose (bool): Whether to randomize the agent pose or not
        custom_bddl (str): If specified, will override internal BDDL
        max_steps (int): Maximum number of steps to take in environment
        skill_kwargs (None or dict): If specified, skill arguments to pass to the internal skill when generating
            trajectories
        use_delta_commands (bool): Whether robot should be using delta commands or not
        visualize_cam_pose (None or 2-tuple): If specified, the relative pose to place the viewer camera
            wrt to the robot's root link. Otherwise, will use a hardcoded default
    """
    def __init__(
            self,
            env,
            eef_z_offset=0.093,
            cab_categories="bottom_cabinet",
            cab_models=None,
            cab_links=None,
            cab_bboxs=None,
            eval_idx=None,
            handle_dist=0.02,
            approach_dist=0.15,
            dist_use_from_handle=True,
            dist_out_from_handle=0.3,
            dist_right_of_handle=-0.1,
            dist_up_from_handle=-0.8,
            z_rot_from_handle=0.0,
            xyz_randomization=None,
            z_rot_randomization=None,
            bbox_randomization=(0, 0, 0),
            randomize_textures=False,
            randomize_cabinet_pose=True,
            randomize_agent_pose=False,
            task_activity_name=None,
            task_obj_bddl_name="cabinet.n.01_1",
            custom_bddl=None,
            max_steps=500,
            skill_kwargs=None,
            use_delta_commands=False,
            visualize_cam_pose=None,
            visualize_skill=False,
            scene_info=None,
            # scene_target_obj_name=None,
            scene_target_parent_obj_name=None,
            scene_target_child_obj_name=None,
            non_target_objs_visual_only=True,
    ):  
        # [OpenCabinetWrapper.__init__] Init parameters:
        # eef_z_offset = 0.18
        # cab_categories = ['top_cabinet', 'bottom_cabinet', 'bottom_cabinet_no_top']
        # cab_models = ['dmwxyl', 'bamfsz', 'vdedzt']
        # cab_links = ['link_1', 'link_1', 'link_0']
        # cab_bboxs = [tensor([0.8236, 0.4751, 0.9036]), tensor([0.8236, 0.4751, 0.9036]), tensor([0.8236, 0.4751, 0.9036])]
        # eval_idx = 1
        # handle_dist = 0.005
        # approach_dist = 0.13
        # dist_use_from_handle = True
        # dist_out_from_handle = 0.5
        # dist_right_of_handle = 0.1
        # dist_up_from_handle = -1.7
        # z_rot_from_handle = 0.0
        # xyz_randomization = [0.03 0.03 0.07]
        # z_rot_randomization = 0.3141592653589793
        # bbox_randomization = [[0.2059029  0.11877306 0.22591069]
        # [0.         0.         0.        ]
        # [0.2059029  0.11877306 0.22591069]]
        # randomize_textures = [True, False, True]
        # randomize_cabinet_pose = False
        # randomize_agent_pose = True
        # task_activity_name = open_cabinet
        # task_obj_bddl_name = cabinet.n.01_1
        # custom_bddl = None
        # max_steps = 500
        # skill_kwargs = {'should_open': True, 'joint_limits': (0.0, 0.7853981633974483), 'n_approach_steps': 15, 'n_converge_steps': 15, 'n_grasp_steps': 1, 'n_articulate_steps': 25, 'n_buffer_steps': 1}
        # use_delta_commands = True
        # visualize_cam_pose = (tensor([-0.2655, -0.3029,  1.8610]), tensor([ 0.3617, -0.2475, -0.5075,  0.7419]))
        # visualize_skill = False
        # scene_info keys = ['resolution', 'scene_graph', 'cam_pose', 'objects']
        # scene_target_obj_name = cabinet_4
        # non_target_objs_visual_only = True
        # Store values
        # 객체의 (x, y, z) 위치에 대해 무작위로 흔들어줄 값의 최대 범위를 설정
        self._xyz_randomization = th.zeros(3) if xyz_randomization is None else th.tensor(xyz_randomization, dtype=th.float)
        # Z축 회전 무작위화
        self._z_rot_randomization = 0.0 if z_rot_randomization is None else z_rot_randomization
        # 캐비닛과 로봇의 pose 무작위화를 할지 여부를 boolean으로 저장
        self._randomize_cabinet_pose = randomize_cabinet_pose
        self._randomize_agent_pose = randomize_agent_pose
        self.skill_kwargs = dict() if skill_kwargs is None else skill_kwargs
        self.eval_idx = eval_idx
        self._current_idx = 0 if self.eval_idx is None else self.eval_idx
        # skill_kwargs = {'should_open': True, 'joint_limits': (0.0, 0.7853981633974483), 'n_approach_steps': 15, 'n_converge_steps': 15, 'n_grasp_steps': 1, 'n_articulate_steps': 25, 'n_buffer_steps': 1}
        # eval_idx = 1
        # _current_idx = 1

        # Values used to randomize other objects
        self.scene_info = scene_info
        self.scene_target_parent_obj_name = scene_target_parent_obj_name
        self.scene_target_child_obj_name = scene_target_child_obj_name
        self.non_target_objs_visual_only = non_target_objs_visual_only
        self.scene_graph = None
        # Scene Graph json load
        if self.scene_info is not None:
            with open(self.scene_info["scene_graph"], "r") as f:
                self.scene_graph = json.load(f)
        self.active_other_obj = dict()
        self._resample_active_obj = False

        # This is the name of lowest object (on the floor) in the subgraph where cabinets are located
        self.obj_root_in_cab_subgraph = None
        if self.scene_info is not None:
            obj_root_in_cab_subgraph = self.scene_graph[self.scene_target_parent_obj_name]["objBeneath"]
            if "wall" in self.scene_graph[self.scene_target_parent_obj_name]["mount"] and "floor" not in self.scene_graph[self.scene_target_parent_obj_name]["mount"]:
                self.obj_root_in_cab_subgraph = None
            elif obj_root_in_cab_subgraph == "floor":
                self.obj_root_in_cab_subgraph = None
            else:
                while self.scene_graph[obj_root_in_cab_subgraph]["objBeneath"] != "floor" and not ("wall" in self.scene_graph[obj_root_in_cab_subgraph]["mount"] and "floor" not in self.scene_graph[obj_root_in_cab_subgraph]["mount"]):
                    obj_root_in_cab_subgraph = self.scene_graph[obj_root_in_cab_subgraph]["objBeneath"]
                self.obj_root_in_cab_subgraph = obj_root_in_cab_subgraph

            for other_obj_name, _ in self.scene_info["objects"].items():
                if other_obj_name == self.scene_target_parent_obj_name:
                    continue
                self.active_other_obj[other_obj_name] = None

        # Store position and orientation of the target object after real2sim
        self.target_parent_obj_position = None
        self.target_parent_obj_orientation = None

        # Whether robot base calculation is wrt handle
        self.dist_use_from_handle = dist_use_from_handle

        # Pick cabinet to load
        cab_categories = [cab_categories] if isinstance(cab_categories, str) else cab_categories
        if cab_models is None:
            cab_models = []
            for cab_category in cab_categories:
                models = get_all_object_category_models(category=cab_category)
                cab_models.append(random.choice(models))
        cab_models = [cab_models] if isinstance(cab_models, str) else cab_models
        self._n_models = len(cab_models)
        # Make sure length of cab categories is the same length as cab_models
        assert len(cab_categories) == self._n_models, "Got mismatch in number of categories vs number of models!"
        if cab_bboxs is None:
            self._default_cab_bboxs = [None] * self._n_models
        else:
            self._default_cab_bboxs = [cab_bboxs] * self._n_models if not isinstance(cab_bboxs[0], Iterable) else cab_bboxs

        if cab_links is None:
            cab_links = [None] * self._n_models
        else:
            cab_links = [cab_links] * self._n_models if isinstance(cab_links, str) else cab_links

        self.target_links = cab_links
        self.robot = env.robots[0]

        # Make sure robot fingers are extra grippy
        from omni.isaac.core.materials import PhysicsMaterial
        gripper_mat = PhysicsMaterial(
            prim_path=f"{self.robot.prim_path}/gripper_mat",
            name="gripper_material",
            static_friction=10.0,
            dynamic_friction=10.0,
            restitution=None,
        )
        for arm, links in self.robot.finger_links.items():
            print(links)
            for link in links:
                for msh in link.collision_meshes.values():
                    msh.apply_physics_material(gripper_mat)
        print("gripper_mat", gripper_mat)
        self.task_obj_bddl_name = task_obj_bddl_name

        # Compute randomization
        bbox_randomization = th.tensor(bbox_randomization, dtype=th.float)
        self._bbox_randomization = bbox_randomization if \
            len(bbox_randomization.shape) == 2 else th.tensor([bbox_randomization] * self._n_models, dtype=th.float)
        self._randomize_textures = [randomize_textures] * self._n_models if \
            isinstance(randomize_textures, bool) else randomize_textures

        # Run super
        super().__init__(env=env, use_delta_commands=use_delta_commands)

        self.cabs = []
        self.target_objs = []
        self.skills = []
        self._default_cab_scales = []
        self._default_target_scales = []
        self._default_cab_poses = []
        self._default_target_poses = []
        self._default_robot_poses = []
        target_parent_obj_bbox_pos, target_parent_obj_bbox_quat = None, None
        if self.scene_info is not None:
            # Assume target object is cab
            og.sim.viewer_camera.set_position_orientation(*self.scene_info["cam_pose"])

            # Load cousin scene objects
            cam_pose = scene_info["cam_pose"]
            cam_pose_mat = OT.pose2mat((th.tensor(cam_pose[0], dtype=th.float), th.tensor(cam_pose[1], dtype=th.float)))
            with og.sim.stopped():
                for obj_name, obj_info in self.scene_info["objects"].items():
                    # print(obj_name, not obj_info["mount"]["floor"])
                    obj = DatasetObject(
                        name=obj_name,
                        category=obj_info["category"],
                        model=obj_info["model"],
                        visual_only=obj_name == self.scene_target_parent_obj_name or self.non_target_objs_visual_only,
                        # visual_only= not (obj_name == self.scene_target_child_obj_name or obj_name == self.scene_target_parent_obj_name),
                        fixed_base=not obj_info["mount"]["floor"],
                        scale=obj_info["scale"]
                    )
                    
                    env.scene.add_object(obj)
                    obj_pos, obj_quat = OT.mat2pose(OT.pose_in_A_to_pose_in_B(
                        pose_A=th.tensor(obj_info["tf_from_cam"], dtype=th.float),
                        pose_A_in_B=cam_pose_mat,
                    ))
                    obj.set_position_orientation(th.tensor(obj_pos, dtype=th.float), th.tensor(obj_quat, dtype=th.float))

                    # If this is the scene target object, make invisible
                    if (obj_name == self.scene_target_parent_obj_name or obj_name == self.scene_target_child_obj_name):
                    # if obj_name == self.scene_target_parent_obj_name :
                        obj.visible = False
                og.sim.step()

            # Record cousin bbox center from cam
            target_parent_obj = env.scene.object_registry("name", self.scene_target_parent_obj_name)
            target_parent_obj_bbox_pos = target_parent_obj.aabb_center
            target_parent_obj_bbox_quat = target_parent_obj.get_position_orientation()[1]

            target_child_obj = env.scene.object_registry("name", self.scene_target_child_obj_name)
            target_child_obj_bbox_pos = target_child_obj.aabb_center
            target_child_obj_bbox_quat = target_child_obj.get_position_orientation()[1]

            # Set this object to be very far away
            target_parent_obj.set_position_orientation(position=th.ones(3) * -300.0)

        for i, (cab_category, cab_model, cab_link, cab_bbox, randomize_tex) in \
                enumerate(zip(cab_categories, cab_models, cab_links, self._default_cab_bboxs, self._randomize_textures)):
            with og.sim.stopped():
                # Move robot out of the scene and place cab at origin
                self.robot.set_position_orientation(position=th.ones(3) * 50.0)

                # Skip the object loading if it is the original target object (i.e.: model)
                cab = DatasetObject(
                    name=f"cabinet{i}",
                    category=cab_category,
                    model=cab_model,
                    bounding_box=cab_bbox,
                    fixed_base=True,
                )

                target_obj = DatasetObject(
                    name=f"pencil_case{i}",
                    category=self.scene_info["objects"][self.scene_target_child_obj_name]["category"],
                    model=self.scene_info["objects"][self.scene_target_child_obj_name]["model"],
                    # visual_only=obj_name == self.scene_target_parent_obj_name or self.scene_target_child_obj_name or self.non_target_objs_visual_only,
                    bounding_box=self.scene_info["objects"][self.scene_target_child_obj_name]["bbox_extent"],
                    fixed_base=not self.scene_info["objects"][self.scene_target_child_obj_name]["mount"]["floor"],
                )
                env.scene.add_object(cab)
                env.scene.add_object(target_obj)

                # Move the cabinet to collision-free space, and make it invisible

                # If we're loading a cousin scene, make sure to set the pose to the correct location relative to the
                # original target object
                if self.scene_info is not None:
                    cab.set_position_orientation(target_parent_obj_bbox_pos, target_parent_obj_bbox_quat)
                    target_obj.set_position_orientation(target_child_obj_bbox_pos, target_child_obj_bbox_quat)
                    og.sim.step_physics()
                    self._default_cab_poses.append(cab.get_position_orientation())
                    self._default_target_poses.append(target_obj.get_position_orientation())

                cab.visible = False
                target_obj.visible = False


                # If we're randomizing the texture, load in per-link materials and bind them
                if randomize_tex:
                    for link_name, link in cab.links.items():
                        mat = MaterialPrim(
                            relative_prim_path=f"{cab._relative_prim_path}/Looks/random_{link_name}_material",
                            name=f"random_material:{link.name}",
                        )
                        mat.load(scene=env.scene)
                        link.material = mat
                
                    for link_name, link in target_obj.links.items():
                        mat = MaterialPrim(
                            relative_prim_path=f"{target_obj._relative_prim_path}/Looks/random_{link_name}_material",
                            name=f"random_material:{link.name}",
                        )
                        mat.load(scene=env.scene)
                        link.material = mat


                    # Render, then force populate all shaders
                    og.sim.render()
                    

                    # Force populate all shaders
                    for link in cab.links.values():
                        link.material.shader_force_populate(render=False)

                    for link in target_obj.links.values():
                        link.material.shader_force_populate(render=False)


            # Get the cabinet joint and link we want to articulate
            if cab_link is None:
                valid_child_link_names = [jnt.child_name for jnt in cab.joints.values()]
                cab_link = random.choice(valid_child_link_names)

            

            # Apply the gripper material so it's extra grippy
            for msh in link.collision_meshes.values():
                msh.apply_physics_material(gripper_mat)

            # # 💥 Coffee box (target_obj) 전체 링크에 적용
            # for link in target_obj.links.values():
            #     for msh in link.collision_meshes.values():
            #         msh.apply_physics_material(gripper_mat)

            link = cab.links[cab_link]

            default_cab_pose = cab.get_position_orientation()

            og.sim.play()

            self.standardize_density_and_friction(obj=cab)
            # Set friction of non-target link to be a large quantity to avoid jittering            
            for joint_name, joint in cab.joints.items():
                if joint_name[2:] != self.target_links[i]:
                    joint.friction = 10.0   # Usually < 0.1


            # # Create the skill
            skill = OpenandGrabSkill(
                robot=self.robot,
                eef_z_offset=eef_z_offset,
                target_obj=cab,
                target_child_obj=target_obj,
                target_link=link,
                handle_dist=handle_dist,
                approach_dist=approach_dist,
                flip_xy_scale_if_not_x_oriented=self.scene_info is None,  # Trust bounding box in cousin scene info
                visualize=visualize_skill,
                visualize_cam_pose=visualize_cam_pose,
            )

            # skill = OpenOrCloseSkill(
            #     robot=self.robot,
            #     eef_z_offset=eef_z_offset,
            #     target_obj=cab,
            #     target_link=link,
            #     handle_dist=handle_dist,
            #     approach_dist=approach_dist,
            #     flip_xy_scale_if_not_x_oriented=self.scene_info is None,  # Trust bounding box in cousin scene info
            #     visualize=visualize_skill,
            #     visualize_cam_pose=visualize_cam_pose,
            # )
            
            # Compute the base pose to set the robot at
            default_robot_pos, default_robot_quat = skill.compute_robot_base_pose(
                dist_use_from_handle=dist_use_from_handle,
                dist_out_from_handle=dist_out_from_handle,
                dist_right_of_handle=dist_right_of_handle,
                dist_up_from_handle=dist_up_from_handle,
            )

            # TODO
            # default_robot_pos[1] 앞 뒤로 +는 Cabinet한테 가까이, -Cabinet한테 멀어짐
            # default_robot_pos = th.tensor([default_robot_pos[0]-0.55, default_robot_pos[1]+0.15, default_robot_pos[2] + 0.65], dtype=th.float)
            
            # default_robot_pos = th.tensor([default_robot_pos[0]-0.40, default_robot_pos[1]+0.1, default_robot_pos[2] + 1.55], dtype=th.float)
            # z_rot_from_handle = 0

            # Multiply default robot pose by default z rot offset
            default_robot_quat = OT.quat_multiply(OT.euler2quat(th.tensor([0, 0, z_rot_from_handle], dtype=th.float)), default_robot_quat)
            default_robot_pose = (default_robot_pos, default_robot_quat)

            self.robot.set_position_orientation(*default_robot_pose)
            self._default_robot_poses.append(default_robot_pose)

            with og.sim.stopped():
                cab.set_position_orientation(position=th.ones(3) * (100 + i * 5.0))
            og.sim.play()

            # Store this cabinet's info
            self.cabs.append(cab)
            self.target_objs.append(target_obj)
            self.skills.append(skill)
            if self.scene_info is None:
                self._default_cab_poses.append(default_cab_pose)
            self._default_cab_scales.append(cab.scale)
            self._default_target_scales.append(target_obj.scale)

            # If not x-oriented, we update the bbox randomization and default bbox values
            if not skill._is_x_oriented:
                if cab_bbox is not None:
                    self._default_cab_bboxs[i] = th.tensor([cab_bbox[1], cab_bbox[0], cab_bbox[2]], dtype=th.float)
                bbox_rand = self._bbox_randomization[i]
                if bbox_rand is not None:
                    self._bbox_randomization[i] = th.tensor([bbox_rand[1], bbox_rand[0], bbox_rand[2]], dtype=th.float)

        # Reset all target objects and robot, then update initial state
        for skill in self.skills:
            skill.reset_target_obj()
        self.robot.reset()
        env.scene.update_initial_state()

        # TODO
        # Override the task
        task_config = deepcopy(env.task_config)
        task_config["type"] = "BehaviorTask"
        task_config["activity_name"] = task_activity_name
        task_config["predefined_problem"] = self.default_bddl if custom_bddl is None else custom_bddl
        task_config["online_object_sampling"] = True
        task_config["termination_config"] = {"max_steps": max_steps}
        env.update_task(task_config=task_config)

        # Reset all target objects and robot, then update initial state
        for skill in self.skills:
            skill.reset_target_obj()
        self.robot.reset()
        env.scene.update_initial_state()
        self.init_state = env.scene.dump_state(serialized=False)

        # Now reset so task is updated properly
        self.reset()

    def reset(self):
        og.sim.play()
        self.env.scene.load_state(self.init_state, serialized=False)
        
        # if self._resample_active_obj:
        self.cabs[self._current_idx].set_position_orientation(
            position=th.ones(3) * (100 + self._current_idx * 5.0),
            orientation=th.tensor([0, 0, 0, 1.0], dtype=th.float),
        )
        self.target_objs[self._current_idx].set_position_orientation(
            position=th.ones(3) * (100 + self._current_idx * 5.0),
            orientation=th.tensor([0, 0, 0, 1.0], dtype=th.float),
        )
        self.cabs[self._current_idx].visible = False
        self.target_objs[self._current_idx].visible = False
        self._current_idx = th.randint(self._n_models, (1,)).item() if self.eval_idx is None else self.eval_idx
        self.cabs[self._current_idx].visible = True
        # self.cabs[self._current_idx].visible = False
        self.target_objs[self._current_idx].visible = True
        

        # TODO
        # Update the task BDDLEntity
        self.env.task.object_scope[self.task_obj_bddl_name].wrapped_obj = self.cabs[self._current_idx]
        # self.env.task.object_scope[self.task_obj_bddl_name].wrapped_obj = self.target_objs[self._current_idx]

        # If we're randomizing bbox, sample new bbox size
        default_bbox = self._default_cab_bboxs[self._current_idx]
        bbox_randomization = self._bbox_randomization[self._current_idx]
        if th.any(bbox_randomization):
            bbox_delta = (th.rand(3) * 2 - 1) * bbox_randomization
            default_scale = self._default_cab_scales[self._current_idx]
            scale_frac = (default_bbox + bbox_delta) / default_bbox
            new_scale = default_scale * scale_frac

            with og.sim.stopped():
                self.cabs[self._current_idx].scale = new_scale
                self.target_objs[self._current_idx].scale = self._default_target_scales[self._current_idx]*scale_frac[2]

        
        # Possibly randomize the materials
        if self._randomize_textures[self._current_idx]:
            link_color = th.rand(3) * 0.25
            handle_color = th.rand(3) * 0.25
            for link in self.cabs[self._current_idx].links.values():
                link.material.diffuse_color_constant = handle_color if "handle" in link.name else link_color
        
        target_objs_color = th.rand(3) * 0.25
        for link in self.target_objs[self._current_idx].links.values():
                if link.material is None:
                    print(f"[⚠️ WARNING] Link '{link.name}' has no material!")
                    continue  # material이 없으면 skip
                try:
                    link.material.diffuse_color_constant = target_objs_color
                except Exception as e:
                    print(f"[❌ ERROR] Failed to set color for link '{link.name}': {e}")
        

        # Call super reset
        super().reset()

        # False
        if self.target_parent_obj_position is not None and self.target_parent_obj_orientation is not None and self.scene_info is None:
            # Also re-randomize the object poses
            self.cabs[self._current_idx].set_position_orientation(*self.randomize_object_pose(
                self.target_parent_obj_position,
                self.target_parent_obj_orientation,
                max_xyz_offset=self._xyz_randomization if self._randomize_cabinet_pose else th.zeros(3),
                max_z_rotation=self._z_rot_randomization if self._randomize_cabinet_pose else 0.0,
            ))

            self.robot.set_position_orientation(*self.randomize_object_pose(
                *self._default_robot_poses[self._current_idx],
                max_xyz_offset=self._xyz_randomization if self._randomize_agent_pose else th.zeros(3),
                max_z_rotation=self._z_rot_randomization if self._randomize_agent_pose else 0.0,
            ))
        else:
            # Also re-randomize the object poses
            # Random Pose 적용안함
            self.cabs[self._current_idx].set_position_orientation(*self.randomize_object_pose(
                *self._default_cab_poses[self._current_idx],
                max_xyz_offset=self._xyz_randomization if self._randomize_cabinet_pose else th.zeros(3),
                max_z_rotation=self._z_rot_randomization if self._randomize_cabinet_pose else 0.0,
            ))


            cab_link = self.cabs[self._current_idx].links[self.target_links[self._current_idx]]  # ← 두 번째 칸
            # Cabinet AABB
            cab_aabb_min, cab_aabb_max = cab_link.aabb
            # Target object AABB
            target_aabb_min, target_aabb_max = self.target_objs[self._current_idx].aabb
            target_y_width = target_aabb_max[1] - target_aabb_min[1]

            padding = 0.05

            allowed_y_min = cab_aabb_min[1] + target_y_width / 2.0 + padding*2
            allowed_y_max = cab_aabb_max[1] - target_y_width / 2.0 - padding*4

            # Target object의 x축 width
            target_x_width = target_aabb_max[0] - target_aabb_min[0]

            # Cabinet AABB로부터 x축에서 안전한 배치 범위 계산
            allowed_x_min = cab_aabb_min[0] + target_x_width / 2.0 + padding*4
            allowed_x_max = cab_aabb_max[0] - target_x_width / 2.0 - padding*4

            # x축도 중앙까지 제한해서 샘플링
            sampled_target_x = random.uniform(allowed_x_min.item(), ((allowed_x_min + allowed_x_max) / 2.0).item())
            sampled_target_y = random.uniform(allowed_y_min.item(), ((allowed_y_min + allowed_y_max) / 2.0).item())
            
            print("📐 [X축 샘플링 범위]")
            print("   ▶ allowed_x_min      =", allowed_x_min.item())
            print("   ▶ allowed_x_max      =", allowed_x_max.item())
            print("   ▶ allowed_x_center   =", ((allowed_x_min + allowed_x_max) / 2.0).item())
            print("   🎯 sampled_target_x  =", sampled_target_x)

            print()

            print("📐 [Y축 샘플링 범위]")
            print("   ▶ allowed_y_min      =", allowed_y_min.item())
            print("   ▶ allowed_y_max      =", allowed_y_max.item())
            print("   ▶ allowed_y_center   =", ((allowed_y_min + allowed_y_max) / 2.0).item())
            print("   🎯 sampled_target_y  =", sampled_target_y)


            # 안전하게 pos, quat 꺼내고 텐서 형태 보장
            pos, quat = self._default_target_poses[self._current_idx]

            # pos가 list of tensor일 경우: stack으로 변환
            if isinstance(pos, list) and all(isinstance(p, th.Tensor) for p in pos):
                pos = th.stack(pos)

            
            # pos, quat이 tensor가 아닐 경우: tensor로 변환
            if not isinstance(pos, th.Tensor):
                pos = th.tensor(pos, dtype=th.float32)
            pos[0] = sampled_target_x
            pos[1] = sampled_target_y

            if not isinstance(quat, th.Tensor):
                quat = th.tensor(quat, dtype=th.float32)

            # 위치 설정
            self.target_objs[self._current_idx].set_position_orientation(pos, quat)

            # self.target_objs[self._current_idx].set_position_orientation(self._default_target_poses[self._current_idx])

            self.robot.set_position_orientation(*self.randomize_object_pose(
                *self._default_robot_poses[self._current_idx],
                max_xyz_offset=self._xyz_randomization if self._randomize_agent_pose else th.zeros(3),
                max_z_rotation=self._z_rot_randomization if self._randomize_agent_pose else 0.0,
                    ))

        step_size = 0.005
        target_obj = self.target_objs[self._current_idx]
        cab_obj = self.cabs[self._current_idx]
        desk_obj = self.env.scene.object_registry("name", "desk_0")
        

        target_obj.keep_still()
        cab_obj.keep_still()
        target_obj.visual_only = False
        cab_obj.visual_only = False
        # desk_obj.keep_still()
        # desk_obj.visual_only = False
        og.sim.step_physics()

        # TODO
        old_state = og.sim.dump_state()

        from omnigibson.object_states import Touching
        # 이미 충돌 중이면 살짝 위로 밀기
        if target_obj.states[Touching].get_value(cab_obj):
            reverse_dir = th.tensor([0, 0, 1.0], dtype=th.float)
            while target_obj.states[Touching].get_value(cab_obj):
                og.sim.load_state(old_state)
                new_pos = target_obj.get_position_orientation()[0] + reverse_dir * step_size
                target_obj.set_position_orientation(position=new_pos)
                old_state = og.sim.dump_state()
                og.sim.step_physics()
                og.sim.step()
                og.sim.render()

        # 아래로 움직이며 충돌할 때까지 붙이기
        step_dir = th.tensor([0, 0, -1.0], dtype=th.float)
        while not target_obj.states[Touching].get_value(cab_obj):
            og.sim.load_state(old_state)
            new_pos = target_obj.get_position_orientation()[0] + step_dir * step_size
            target_obj.set_position_orientation(position=new_pos)
            old_state = og.sim.dump_state()
            og.sim.step_physics()
            og.sim.step()
            og.sim.render()

        # 마지막 1스텝 back해서 살짝 위로 되돌리기
        og.sim.load_state(old_state)
        final_pos = target_obj.get_position_orientation()[0] - step_dir * step_size
        target_obj.set_position_orientation(position=final_pos)


        # Adjust z-value only based on scene graph
        # Enable physics for objects on the same sub-graph
        if self.obj_root_in_cab_subgraph and self.scene_graph and self.active_other_obj[self.obj_root_in_cab_subgraph]:
            def _adjust_z_by_graph(root_name):
                # First adjust z of root, and enable physics
                obj_beneath_name = self.scene_graph[root_name]["objBeneath"]
                if root_name == self.scene_target_parent_obj_name:
                    # cab_center_x, cab_center_y, cab_center_z = self.cabs[self._current_idx].aabb_center
                    cab_center_x, cab_center_y, cab_center_z = self.cabs[self._current_idx].get_position_orientation()[0]
                    # adjust objects beneath cab
                    if obj_beneath_name == "floor":
                        lower_corner, _ = self.cabs[self._current_idx].aabb
                        cab_low_z = lower_corner[-1]
                        translate_vec = [0, 0, -cab_low_z]
                    else:
                        obj_beneath = self.active_other_obj[obj_beneath_name]
                        lower_corner, _ = self.cabs[self._current_idx].aabb
                        cab_low_z = lower_corner[-1]
                        _, upper_corner = obj_beneath.aabb
                        up_z = upper_corner[-1]
                        translate_vec = [0, 0, up_z - cab_low_z]
                    cab_new_center = (cab_center_x + translate_vec[0], cab_center_y + translate_vec[1], cab_center_z + translate_vec[2])
                    self.cabs[self._current_idx].set_position_orientation(position=th.tensor(cab_new_center, dtype=th.float))
                    robot_center_x, robot_center_y, robot_center_z = self.robot.get_position_orientation()[0]
                    robot_new_center = (robot_center_x + translate_vec[0], robot_center_y + translate_vec[1], robot_center_z + translate_vec[2])
                    self.robot.set_position_orientation(position=th.tensor(robot_new_center, dtype=th.float))
                else:
                    # When we randomly sample non-tar obj, if the obj is not imported, we directly return
                    if self.active_other_obj[root_name] is None:
                        return
                    # Otherwise, adjust z of subgraph rooted at this non-tar obj
                    center_x, center_y, center_z = self.active_other_obj[root_name].get_position_orientation()[0]
                    if obj_beneath_name == "floor":
                        lower_corner, _ = self.active_other_obj[root_name].aabb
                        low_z = lower_corner[-1]
                        translate_vec = [0, 0, -low_z]
                    else:
                        obj_beneath = self.active_other_obj[obj_beneath_name]
                        lower_corner, _ = self.active_other_obj[root_name].aabb
                        low_z = lower_corner[-1]
                        _, upper_corner = obj_beneath.aabb
                        up_z = upper_corner[-1]
                        translate_vec = [0, 0, up_z - low_z]
                    new_center = (center_x + translate_vec[0], center_y + translate_vec[1], center_z + translate_vec[2])
                    self.active_other_obj[root_name].set_position_orientation(th.tensor(new_center, dtype=th.float))

                for _ in range(3):
                    og.sim.step()
                for _ in range(3):
                    og.sim.render()

                # Termination condition: 
                if self.scene_graph[root_name]["objOnTop"] is None or len(self.scene_graph[root_name]["objOnTop"]) == 0:
                    return

                # Adjust all objects on top 
                for obj_on_top_name in self.scene_graph[root_name]["objOnTop"]:
                    _adjust_z_by_graph(obj_on_top_name)
           
            _adjust_z_by_graph(self.obj_root_in_cab_subgraph)
        

        if self.scene_info is None:
            # Set the camera to visualize
            self.skills[self._current_idx].set_camera_to_visualize()

        self.cabs[self._current_idx].keep_still()
        self.target_objs[self._current_idx].keep_still()
        # Step and grab new obs
        for _ in range(5):
            og.sim.step()
        
        for _obj in self.scene.objects:
            if ToggledOn in _obj.states:
                _obj.states[ToggledOn].link.visible = False

        for _ in range(100):
            og.sim.render()

        obs, _ = self.env.get_obs()
        self._resample_active_obj = False   # Finish sampling activate obj, set it back to False
        return obs

    def set_cabinet_idx(self, idx):
        """
        Sets the internal cabinet index @idx. This automatically resets the environment and sets the internal
        @self.eval_idx.

        Args:
            idx (int): The index in @cab_models array to use for evaluating this environment
        """
        assert idx < self._n_models, \
            f"Got invalid idx for set_cabinet_idx -- must be positive integer less than {self._n_models}"
        self.eval_idx = idx
        self._resample_active_obj = True

    @staticmethod
    def randomize_object_pose(default_pos, default_quat, max_xyz_offset=(0, 0, 0), max_z_rotation=0.0):
        """
        Randomizes the pose given @default_pos and @default_quat, based on max perurbations @max_xyz_offset and
        @max_z_rotation

        Args:
            default_pos (3-array): (x,y,z) position to perturb
            default_quat (4-array): (x,y,z,w) quaternion orientation to perturb
            max_xyz_offset (3-array): (x,y,z) maximum perturbation to sample
            max_z_rotation (float): maximum z-rotation to sample

        Returns:
            2-tuple:
                - torch.tensor: (x,y,z) perturbed position
                - torch.tensor: (x,y,z,w) perturbed quaternion
        """
        # Sample point using radius as constraint
        max_xyz_offset = th.tensor(max_xyz_offset, dtype=th.float)
        pos_offset = th.rand(3) * (2.0 * max_xyz_offset) - max_xyz_offset
        rot_z_offset = OT.euler2mat(th.tensor([0.0, 0.0, th.rand(1).item() * 2.0 * max_z_rotation - max_z_rotation], dtype=th.float))
        new_pos = default_pos + pos_offset
        new_quat = OT.mat2quat(rot_z_offset @ OT.quat2mat(default_quat))
        return new_pos, new_quat

    @staticmethod
    def standardize_density_and_friction(obj, density=200.0, friction=0.025):
        """
        Standardizes density and friction for obj @obj across all of its links and joints with values
        @density and @friction

        Args:
            obj (BaseObject): Object to modify density and friction
            density (float): Density to set
            friction (float): Friction to set
        """
        for lnk in obj.links.values():
            lnk.density = density
        for jnt in obj.joints.values():
            jnt.friction = friction

    @property
    def default_bddl(self):
        """
        Returns:
            str: The default BDDL for this task wrapper
        """
        return """
        (define (problem open_cabinet-0)
            (:domain omnigibson)

            (:objects
                cabinet.n.01_1 - cabinet.n.01
                agent.n.01_1 - agent.n.01
            )

            (:init
                (inroom cabinet.n.01_1 none)
                (not
                    (open cabinet.n.01_1)
                )
            )

            (:goal
                (open cabinet.n.01_1)
            )
        )
        """

    @property
    def solve_steps(self):
        # We don't need the final move out motion for solving this task
        # return [e for e in self.skill.steps][:-1]
        return [e for e in self.skill.steps]

    def get_skill_and_kwargs_at_step(self, solve_step):
        return self.skill, solve_step, self.skill_kwargs, True

    @property
    def skill(self):
        return self.skills[self._current_idx]
