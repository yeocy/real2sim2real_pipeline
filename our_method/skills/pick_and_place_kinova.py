import omnigibson as og
from omnigibson.controllers import OperationalSpaceController, InverseKinematicsController, MultiFingerGripperController
from omnigibson.objects import PrimitiveObject
import omnigibson.utils.transform_utils as OT
from omnigibson.utils.sampling_utils import raytest_batch
import omnigibson.lazy as lazy
from our_method.skills.skill_base import ManipulationSkill
import torch as th
from enum import IntEnum


# Specific stage of the skill
class OpenOrCloseStep(IntEnum):
    CABINET_APPROACH = 0
    CABINET_CONVERGE = 1
    CABINET_GRASP = 2
    CABINET_ARTICULATE = 3
    CABINET_UNGRASP = 4
    CABINET_RETREAT = 5
    # ROBOT_RETURN_TO_INITIAL = 6
    # TARGET_APPROACH = 7
    # TARGET_CONVERGE = 8
    # TARGET_GRASP = 9
    # TARGET_UP = 10
    # TARGET_PLACE_1 = 11
    # TARGET_PLACE_2 = 12
    # TARGET_UNGRASP = 13
    # TARGET_RELEASE = 14


class PickAndPlaceSkill(ManipulationSkill):
    """
    Class for opening / closing an articulated object. It is assumed the articulated object has a handle to grasp, which
    will automatically be detected.
    """

    def __init__(
            self,
            robot,
            target_obj,
            target_child_obj,
            eef_z_offset=0.093,
            handle_dist=0.02,
            handle_offset=None,
            approach_dist=0.2,
            flip_xy_scale_if_not_x_oriented=False,
            visualize=True,
            visualize_cam_pose=None,
    ):
        """
        Args:
            robot (ManipulationRobot): Robot on which to deploy the skill
            target_obj (BaseObject): Articulated object to open / close
            target_link (RigidPrim): Which link of @target_obj to articulate
            eef_z_offset (float): Distance in the robot's EEF z-direction specifying distance to its actual grasping
                location for its assumed parallel jaw gripper
            handle_dist (float): Distance into a detected handle to move the robot EEF
            handle_offset (None or 3-array): If specified, (x,y,z) offset in the local handle frame (where x faces outward,
                y faces rightward, z faces upward) when computing the grasping point
            approach_dist (float): Distance from the front of the handle when approaching for a grasp
            flip_xy_scale_if_not_x_oriented (bool): If True, will flip the object's xy scale if it is not x-orinted.
                This is useful if the target object's bounding box was set programmatically with the expectation that
                the bbox x-value was assumed to correspond to the dimension facing the front of the cabinet.
            visualize (bool): Whether to visualize this skill or not
            visualize_cam_pose (None or 2-tuple): If specified, the relative pose to place the viewer camera
                wrt to the robot's root link. Otherwise, will use a hardcoded default
        """
        # Make sure we have a valid link and that the target object is fixed
        # assert target_obj.fixed_base, \
        #     f"Can only use {self.__class__.__name__} with a target_obj that has fixed_base=True!"

        # Store target obj information
        self._target_obj = target_obj
        self._target_child_obj = target_child_obj
        self._handle_dist = handle_dist
        self._handle_offset = th.zeros(3) if handle_offset is None else th.tensor(handle_offset, dtype=th.float)
        self._approach_dist = approach_dist
        # self._approach_dist = 0.5
        self._flip_xy_scale_if_not_x_oriented = flip_xy_scale_if_not_x_oriented

        # Other info that will be filled in later
        self._marker = None                         # PrimitiveObject
        self._default_scale = None                  # Scale when this skill is initialized
        self._default_obj_to_grasp_pos = None       # Relative position of the object to the grasp position wrt the default scale
        self._obj_z_rot_offset = None               # (3, 3)-array
        self._is_x_oriented = None                  # bool
        self._is_vertical_handle = None             # bool
        self._target_joint = None                   # JointPrim
        self._joint_axis_idx = None                 # int
        self._joint_rel_mat = None                  # (3, 3)-array
        self._joint_to_handle_pos = None            # 3-array
        self._joint_to_approach_pos = None          # 3-array
        self._joint_to_approach_target_pos = None
        self._approach_idx = None                   # {0, 1}
        self._approach_sign = None                  # {-1, 1}
        self._link_to_grasp_pos = None              # 3-array
        self._update_grasp_pose = None             # Lambda function that internally updates joint_to_handle/approach_pos based on current obj scale
        self.target_step = False

        # Call super
        super().__init__(
            robot=robot,
            eef_z_offset=eef_z_offset,
            visualize=visualize,
            visualize_cam_pose=visualize_cam_pose,
        )

    def initialize(self):
        # Store the current state of the simulator so we can restore it later
        state = og.sim.dump_state(serialized=False)

        # Run sanity checks to make sure robot is using expected action type
        # The arm must be using OSC, with absolute_pose values
        arm_controller = self._robot.controllers[f"arm_{self._robot.default_arm}"]
        eef_controller = self._robot.controllers[f"gripper_{self._robot.default_arm}"]
        # print("=== Control Limits ===")
        # print("Position:", eef_controller._control_limits)
        # print("dof_idx:", eef_controller.dof_idx)
        # print("Position:", eef_controller._control_limits["position"])
        # print("Velocity:", eef_controller._control_limits["velocity"])
        # print("Effort:", eef_controller._control_limits["effort"])
        # print("Has Limit:", eef_controller._control_limits["has_limit"])
        assert (isinstance(arm_controller, OperationalSpaceController) or
                isinstance(arm_controller, InverseKinematicsController)), \
            f"Skill {self.__class__.__name__} requires OSC or IK controller for arm!"
        assert isinstance(eef_controller, MultiFingerGripperController), f"Skill {self.__class__.__name__} requires MultiFingerGripper controller for EEF!"
        # assert eef_controller._mode == "binary", f"Skill {self.__class__.__name__} requires mode 'binary' for EEF controller!"
        assert not eef_controller._inverted, f"Skill {self.__class__.__name__} requires inverted=False for EEF controller!"

        # Move the target object into space and check orientation
        self._target_obj.set_position_orientation(th.ones(3) * -100.0, th.tensor([0, 0, 0, 1.0], dtype=th.float))

        self._target_obj.keep_still()
        og.sim.step()
        original_aabb_extent = self._target_obj.aabb_extent
        og.sim.step()
        new_aabb_extent = self._target_obj.aabb_extent
        aabb_extent_diff = new_aabb_extent - original_aabb_extent
        self._is_x_oriented = aabb_extent_diff[0] > aabb_extent_diff[1]
        ##################################################################################################

        # XY 비율 보정
        if not self._is_x_oriented and self._flip_xy_scale_if_not_x_oriented:
            # Flip xy scale
            with og.sim.stopped():
                xy_extent_ratio = original_aabb_extent[1] / original_aabb_extent[0]
                obj_scale = self._target_obj.scale
                self._target_obj.scale = obj_scale * th.tensor([xy_extent_ratio, 1 / xy_extent_ratio, 1.0], dtype=th.float)

        # Z축 회전 보정 행렬
        # If Y-oriented, we rotate the cabinet by 90 deg wrt the Z-axis
        self._obj_z_rot_offset = OT.quat2mat(th.tensor([0, 0, 0, 1.0], dtype=th.float) if self._is_x_oriented else th.tensor([0, 0, 0.707, 0.707], dtype=th.float))

        # Stop, make the target object disable gravity only, then set it into the sky
        with og.sim.stopped():
            self._target_obj.disable_gravity()

        # Move target obj into space
        obj_pos_offset = th.ones(3) * 200.0
        self._target_obj.set_position_orientation(obj_pos_offset, OT.mat2quat(self._obj_z_rot_offset))
        og.sim.step()

        # 기본 스케일 저장
        self._default_scale = self._target_obj.scale

        # Reset the target object to be normal facing
        self._target_obj.set_position_orientation(orientation=th.tensor([0, 0, 0, 1.0], dtype=th.float))

        # 초기 grasp 기준 위치를 object 중심에서 바로 위쪽(조정가능)으로 설정
        obj_pos, obj_ori = self._target_obj.get_position_orientation()
        self._default_scale = self._target_obj.scale

        # 예를 들어 위쪽 (z축 10cm) 오프셋을 default grasp 위치로 한다면
        self._default_obj_to_approach_pos = th.tensor([0.0, 0.0, 0.2], dtype=th.float32)
        self._default_obj_to_convergence_pos = th.tensor([0.0, 0.0, 0.0], dtype=th.float32)

        # 회전 오프셋은 필요없으면 Identity
        self._obj_z_rot_offset = th.eye(3)


        # Set approach axis info
        self._approach_idx = 0 if self._is_x_oriented else 1
        self._approach_sign = 1 if self._is_x_oriented else -1

        def pose_updater():
            # Get updated scale ratio
            scale = self._target_obj.scale
            scale_frac = scale / self._default_scale

            # Updated grasp position = (초기 오프셋) * 스케일 조정
            grasp_approach_pos_canonical = self._obj_z_rot_offset.T @ (self._default_obj_to_approach_pos) * scale_frac
            grasp_convergence_pos_canonical = self._obj_z_rot_offset.T @ (self._default_obj_to_convergence_pos) * scale_frac

            # object의 현재 position을 가져와서 적용
            obj_pos, obj_ori = self._target_obj.get_position_orientation()
            target_obj_pos, obj_ori = self._target_child_obj.get_position_orientation()

            self.obj_to_hand_approach = grasp_approach_pos_canonical + obj_pos
            self.obj_to_hand_convergence = grasp_convergence_pos_canonical + obj_pos
            print(f"Joint to Handle Position: {self._joint_to_handle_pos}")
            self.obj_to_target_approach = grasp_approach_pos_canonical + target_obj_pos
            self.obj_to_target_convergence = grasp_convergence_pos_canonical + target_obj_pos
            
            # self._joint_to_approach_pos_2 = grasp_pos_canonical + obj_pos + self._approach_dist / 2
            # print(f"Joint to Approach Position 2: {self._joint_to_approach_pos_2}")

            # child object도 업데이트
            child_pos, _ = self._target_child_obj.get_position_orientation()
            self._joint_to_approach_target_pos = child_pos


        # grasp pose 계산 함수 정의 및 실행
        self._update_grasp_pose = pose_updater
        self._update_grasp_pose()


        ##################################################################################################

        # Visualize with marker if requested
        if self._visualize:
            self._marker = PrimitiveObject(
                name=f"insert_skill_{self._target_obj.name}_marker",
                primitive_type="Sphere",
                visual_only=True,
                radius=0.01,
                rgba=[0, 1.0, 1.0, 1.0],
            )
            self._scene.add_object(self._marker)

        ##################################################################################################

        # Restore state
        og.sim.load_state(state, serialized=False)
        self._initial_eef_pos, self._initial_eef_quat = self._robot.get_relative_eef_pose(mat=False)


    def compute_grasp_pose(self, joint_to_grasp_pos, delta_jnt_val=0.0, return_mat=False):
        """
        Computes the grasp pose for the desired handle attached to @self._target_link. Note: Assumes a parallel jaw
        gripper with its local orientation such that Z points out of the EEF and Y points in the direction of the
        jaw articulation

        Args:
            joint_to_grasp_pos (torch.tensor): (x,y,z) relative position of the desired grasping point wrt the joint frame
            delta_joint_val (float): If specified, the desired delta_joint value for computing the grasp pose
            return_mat (bool): Whether to return the orientation as a 3x3 matrix or a 4-array quaternion

        Returns:
            2-tuple:
                - torch.tensor: (x,y,z) global handle grasping position
                - torch.tensor: (x,y,z,w) global handle grasping quaternion or (3,3)-shaped orientation matrix
        """
        # Compute relevant state
        # 현재 link pos, quat 받아옴
        obj_pos, obj_quat = self._target_obj.get_position_orientation()
        obj_mat = OT.quat2mat(obj_quat)

        # Assume x points out from the cabinet, y points right, z points up
        # Then transform (in the drawer's local frame) to have robot gripper point towards it is to rotate it -90 degrees wrt
        # to the Y-axis, and then optionally 90 deg wrt the X axis depending on if the drawer is horizontal or not
        # 문 손잡이가 수직이면 self._is_vertical_handle = True  회전 X
        # 문 손잡이가 수평이면 self._is_vertical_handle = False 회전 O
        gripper_yaw = 0.0 if self._is_vertical_handle else -th.pi / 2
        # print(OT.euler2mat(th.tensor([gripper_yaw, 0, 0], dtype=th.float)) @ OT.euler2mat(th.tensor([0, -th.pi / 2, 0], dtype=th.float)))
        if self.target_step:
            # grasp_mat = self._obj_z_rot_offset.T @ link_mat @ OT.euler2mat(th.tensor([-th.pi / 2, 0, -th.pi / 2], dtype=th.float))
            # grasp_mat = self._obj_z_rot_offset.T @ link_mat @ OT.euler2mat(th.tensor([gripper_yaw, 0, 0], dtype=th.float)) @ OT.euler2mat(th.tensor([0, 0, 0], dtype=th.float))
            # grasp_mat = self._obj_z_rot_offset.T @ link_mat @ OT.euler2mat(th.tensor([0, 0, -th.pi / 2], dtype=th.float))

 
            # 기존 grasp_mat 계산 뒤에 추가 회전
            # [th.pi / 2, 0, 0], [0, 0, th.pi / 2]
            # grasp_mat = self._obj_z_rot_offset.T @ link_mat @ OT.euler2mat(th.tensor([th.pi / 2, 0, 0], dtype=th.float)) @ OT.euler2mat(th.tensor([0, th.pi / 2, 0], dtype=th.float))
            # grasp_mat = self._obj_z_rot_offset.T @ link_mat @ OT.euler2mat(th.tensor([0, 0, th.pi / 2], dtype=th.float)) @ OT.euler2mat(th.tensor([0, th.pi / 2, 0], dtype=th.float))
            # grasp_mat = self._obj_z_rot_offset.T @ link_mat @ OT.euler2mat(th.tensor([0, 0, th.pi / 2], dtype=th.float)) @ OT.euler2mat(th.tensor([0, 0, th.pi / 2], dtype=th.float))
            # grasp_mat = self._obj_z_rot_offset.T @ link_mat @ OT.euler2mat(th.tensor([0, 0, th.pi / 2], dtype=th.float)) @ OT.euler2mat(th.tensor([th.pi / 2, 0, 0], dtype=th.float))
            # grasp_mat = self._obj_z_rot_offset.T @ link_mat @ OT.euler2mat(th.tensor([0, th.pi / 2, 0], dtype=th.float)) @ OT.euler2mat(th.tensor([th.pi / 2, 0, 0], dtype=th.float))
            # grasp_mat = self._obj_z_rot_offset.T @ link_mat @ OT.euler2mat(th.tensor([0, th.pi / 2, 0], dtype=th.float)) @ OT.euler2mat(th.tensor([0, -th.pi / 2, 0], dtype=th.float))
            # grasp_mat = self._obj_z_rot_offset.T @ link_mat @ OT.euler2mat(th.tensor([0, th.pi / 2, 0], dtype=th.float)) @ OT.euler2mat(th.tensor([0, 0, th.pi / 2], dtype=th.float))
            # grasp_mat = self._obj_z_rot_offset.T @ link_mat @ OT.euler2mat(th.tensor([th.pi / 2, 0, 0], dtype=th.float)) @ OT.euler2mat(th.tensor([0, 0, th.pi / 2], dtype=th.float))
            # grasp_mat = self._obj_z_rot_offset.T @ link_mat @ OT.euler2mat(th.tensor([th.pi / 2, 0, 0], dtype=th.float)) @ OT.euler2mat(th.tensor([0, th.pi / 2, 0], dtype=th.float))
            # grasp_mat = self._obj_z_rot_offset.T @ link_mat @ OT.euler2mat(th.tensor([th.pi / 2, 0, 0], dtype=th.float)) @ OT.euler2mat(th.tensor([th.pi / 2, 0, 0], dtype=th.float))
            # grasp_mat = self._obj_z_rot_offset.T @ link_mat @ OT.euler2mat(th.tensor([0, 0, 0], dtype=th.float)) @ OT.euler2mat(th.tensor([th.pi / 2, 0, 0], dtype=th.float))
            # grasp_mat = self._obj_z_rot_offset.T @ link_mat @ OT.euler2mat(th.tensor([0, 0, 0], dtype=th.float)) @ OT.euler2mat(th.tensor([0, th.pi / 2, 0], dtype=th.float))
            # grasp_mat = self._obj_z_rot_offset.T @ link_mat @ OT.euler2mat(th.tensor([0, 0, 0], dtype=th.float)) @ OT.euler2mat(th.tensor([0, 0, th.pi / 2], dtype=th.float))
            # grasp_mat = self._obj_z_rot_offset.T @ link_mat @ OT.euler2mat(th.tensor([0, 0, 0], dtype=th.float)) @ OT.euler2mat(th.tensor([0, 0, th.pi], dtype=th.float))
            # grasp_mat = self._obj_z_rot_offset.T @ link_mat @ OT.euler2mat(th.tensor([0, 0, 0], dtype=th.float)) @ OT.euler2mat(th.tensor([0, -th.pi, 0], dtype=th.float)) # 강추 
            # grasp_mat = self._obj_z_rot_offset.T @ link_mat @ OT.euler2mat(th.tensor([0, 0, 0], dtype=th.float)) @ OT.euler2mat(th.tensor([th.pi, 0, 0], dtype=th.float))
            grasp_mat = self._obj_z_rot_offset.T @ obj_mat @ OT.euler2mat(th.tensor([0, 0, -th.pi / 2], dtype=th.float)) @ OT.euler2mat(th.tensor([0, -th.pi, 0], dtype=th.float))
            # grasp_mat = self._obj_z_rot_offset.T @ link_mat @ OT.euler2mat(th.tensor([th.pi, 0, 0], dtype=th.float)) @ OT.euler2mat(th.tensor([0, -th.pi, 0], dtype=th.float))

        else:
            # grasp_mat = self._obj_z_rot_offset.T @ obj_mat @ OT.euler2mat(th.tensor([gripper_yaw, 0, 0], dtype=th.float)) @ OT.euler2mat(th.tensor([0, -th.pi / 2, 0], dtype=th.float))
            grasp_mat = self._obj_z_rot_offset.T @ obj_mat @ OT.euler2mat(th.tensor([0, 0,  -th.pi], dtype=th.float)) @ OT.euler2mat(th.tensor([0, -th.pi, 0], dtype=th.float))
        
        # self._obj_z_rot_offset = OT.quat2mat(th.tensor([0, 0, 0, 1.0], dtype=th.float) if self._is_x_oriented else th.tensor([0, 0, 0.707, 0.707], dtype=th.float))
        # print(OT.euler2mat(th.tensor([th.pi, -th.pi/2, 0]))
        # exit()
            

        # new_grasp_mat_global_frame : 로봇이 도착해야하는 손목의 matrix
        # new_grasp_pos_parent_frame : 타겟 link의 local 좌표계 기준에서의 grasp 위치

        # grasp 위치(포지션)를 global 좌표계로 변환
        new_grasp_pos_global_frame = joint_to_grasp_pos
        new_grasp_mat_global_frame = OT.euler2mat(th.tensor([0.0, 0.0, 0.0])) @ grasp_mat

        return new_grasp_pos_global_frame, (new_grasp_mat_global_frame if return_mat else OT.mat2quat(new_grasp_mat_global_frame))

    def compute_robot_base_pose(self, dist_use_from_handle=True, dist_out_from_handle=0.2, dist_right_of_handle=-0.2, dist_up_from_handle=-0.8):
        """
        Computes the pose to set the robot's base at given a relative distance from @self._target_link's handle. Note
        that this will automatically take into account @self._target_obj's orientation (with respect to global frame
        AND handle-forward convention) such that the outputted robot orientation will be facing the target object's
        articulated face.

        Args:
            dist_use_from_handle (bool): Whether use distance from handle (Otherwise, use distance from base)
            dist_out_from_handle (float): Distance orthogonal to the front of the handle
            dist_right_of_handle (float): Distance to the right of the handle, when viewed from the front
            dist_up_from_handle (float): Distance upwards from the handle

        Returns:
            2-tuple:
                - torch.tensor: (x,y,z) global robot base position
                - torch.tensor: (x,y,z,w) global robot base quaternion
        """
        if dist_use_from_handle:
            robot_base_pos_offset = th.zeros(3)
            robot_base_pos_offset[self._approach_idx] = dist_out_from_handle * self._approach_sign
            robot_base_pos_offset[1 - self._approach_idx] = dist_right_of_handle
            robot_base_pos_offset[2] = dist_up_from_handle
            # Do the reverse rotation to offset cabinet rotation
            robot_base_quat_offset = OT.mat2quat(self._obj_z_rot_offset.T @ OT.euler2mat(th.tensor([0, 0, th.pi], dtype=th.float)))

            # Convert to global frame
            obj_pos, obj_quat = self._target_obj.get_position_orientation()
            obj_mat = OT.quat2mat(obj_quat)
            grasp_pos, _ = self.compute_grasp_pose(joint_to_grasp_pos=obj_pos)
            robot_base_pos = grasp_pos + obj_mat @ robot_base_pos_offset
            robot_base_quat = OT.mat2quat(obj_mat @ OT.quat2mat(robot_base_quat_offset))
        else:
            target_obj_aabb = self._target_obj.aabb_extent
            robot_base_pos_offset = th.zeros(3)
            robot_base_pos_offset[self._approach_idx] = (target_obj_aabb[self._approach_idx] / 2 + dist_out_from_handle) * self._approach_sign
            robot_base_pos_offset[1 - self._approach_idx] = dist_right_of_handle
            robot_base_pos_offset[2] = dist_up_from_handle

            target_ori_mat = OT.quat2mat(self._target_obj.get_orientation())
            robot_base_pos = self._target_obj.aabb_center + target_ori_mat @ robot_base_pos_offset
            robot_base_quat = OT.mat2quat(target_ori_mat @ self._obj_z_rot_offset.T @ OT.euler2mat([0, 0, th.pi]))

        return robot_base_pos, robot_base_quat

    def compute_current_subtrajectory(
            self,
            step,
            should_open=True,
            joint_limits=None,
            n_approach_steps=150,
            n_converge_steps=200,
            n_grasp_steps=20,
            n_articulate_steps=200,
            n_buffer_steps=5,
            max_open_val=None,
            grasp_override_val=None,
            maintain_current_orientation=False,
            enable_finetune_trajopt=True,
    ):
        """
        Computes the subtrajectory for executing the next substep of the skill.

        NOTE: Assumes joint's lower limit --> Close, upper limit --> Open

        Args:
            step (OpenOrCloseStep): Which step to compute subtrajectory for
            should_open (bool): Whether the desired skill is Open or Close
            joint_limits (None or 2-tuple): If specified, the (min, max) limits of the joint defining the range
                to be articulated. If None, will infer from @self._target_joint's upper / lower limits
            n_approach_steps (int): Number of steps for robot to move to approach pose
            n_converge_steps (int): Number of steps for robot to converge towards handle pose
            n_grasp_steps (int): Number of steps for robot to un/grasp handle
            n_articulate_steps (int): Number of steps for robot to open/close and articulate the joint
            n_buffer_steps (int): The number of steps to include at the end of the subtrajectory repeating the
                final action
            max_open_val (None or float): If specified, and if in step OpenOrCloseStep.ARTICULATE, this specifies
                the maximum joint value (either in m or rad) when opening the object. Otherwise, will infer the value
                directly from the upper joint limit.
            grasp_override_val (None or bool): If set, override grasping value to send
            maintain_current_orientation (bool): Whether to maintain the current orientation or plan an optimal
                orientation as well
            enable_finetune_trajopt (bool): Whether to enable timing reparameterization for a smoother trajectory

        Returns:
            2-tuple:
                - torch.tensor: (T, D)-shaped array where D-length actions are stacked to form an T-length
                    subtrajectory action sequence to deploy in an environment
                - torch.tensor: (T, D)-shaped array where D-length actions are stacked to form an T-length
                    subtrajectory nullspace action sequence to deploy in an environment
        
        
        step	수행할 스킬 단계 (APPROACH, CONVERGE, GRASP 등)
        should_open	문을 여는 동작인지 (True) 닫는 동작인지 (False) 
        joint_limits	관절의 허용된 값 범위 (기본은 해당 링크의 설정 값)    (0.0, 0.7853981633974483)
        n_*_steps	각 동작 단계에서 몇 스텝 동안 수행할지 (e.g., approach, articulate 등)    (15, 15, 1, 25, 1)
        max_open_val	문을 열 때 최대 열림 값 (없으면 joint limit 사용)       None
        grasp_override_val	그립 값을 강제로 지정할지 여부      None
        maintain_current_orientation	기존 EEF 자세를 유지할지 여부       False
        enable_finetune_trajopt	(사용되지 않음) 시간 최적화 여부 (현재는 트라젝토리 생성 후 재보정 미사용)      True

        
        """
        # 5 steps:
        # (1) Move to approach pose
        # (2) Converge to the handle pose
        # (3) Grasp the handle
        # (4) Articulate (open / close) the link
        # (5) Release grasp

        # Update grasp poses
        self._update_grasp_pose()
        # If visualize, set camera to visualize:
        if self._visualize:
            self.set_camera_to_visualize()

        no_op = False
        joint_to_grasp_pos = None
        delta_jnt_vals = None

        # grasp = True
        #         # (1) Move to approach pose
        # if step == OpenOrCloseStep.CABINET_APPROACH:
        #     # n_steps = n_approach_steps
        #     # joint_to_grasp_pos = self._joint_to_approach_pos
        #     # grasp = False
        #     self.target_step = True
        #     n_steps = n_approach_steps
        #     # self._joint_to_approach_target_pos[0] = -0.1383
        #     # self._joint_to_approach_target_pos[1] = -5.6
        #     # self._joint_to_approach_target_pos[2] = 1.1141
            
        #     # self._joint_to_approach_pos[0] -= 0.4
        #     # self._joint_to_approach_pos[2] -= 0.4
        #     joint_to_grasp_pos = self._joint_to_approach_pos
        #     grasp = False
        
        # # (3) Grasp the handle
        # elif step == OpenOrCloseStep.CABINET_CONVERGE:
        #     n_steps = n_grasp_steps
        #     no_op = True
        #     grasp = True
        
        # # (5) Release grasp
        # elif step == OpenOrCloseStep.CABINET_GRASP:
        #     n_steps = n_grasp_steps
        #     no_op = True
        #     grasp = False


        self.target_step = False
        self.target_approach = 0.0
        # (1) Move to approach pose
        if step == OpenOrCloseStep.CABINET_APPROACH:
            n_steps = n_approach_steps
            joint_to_grasp_pos = self.obj_to_hand_approach
            print("joint_to_grasp_pos: ", joint_to_grasp_pos)
            grasp = False

        # (2) Approach the handle
        elif step == OpenOrCloseStep.CABINET_CONVERGE:
            n_steps = n_converge_steps
            joint_to_grasp_pos = self.obj_to_hand_convergence
            grasp = False

        # (3) Grasp the handle
        elif step == OpenOrCloseStep.CABINET_GRASP:
            n_steps = n_grasp_steps
            no_op = True
            grasp = True

        # # (4) Open the link
        elif step == OpenOrCloseStep.CABINET_ARTICULATE:
            n_steps = n_approach_steps
            joint_to_grasp_pos = self.obj_to_target_approach
            print("joint_to_grasp_pos: ", joint_to_grasp_pos)
            grasp = True
               # # (4) Open the link
        elif step == OpenOrCloseStep.CABINET_UNGRASP:
            n_steps = n_approach_steps
            joint_to_grasp_pos = self.obj_to_target_convergence
            print("joint_to_grasp_pos: ", joint_to_grasp_pos)
            grasp = True


        # (5) Release grasp
        elif step == OpenOrCloseStep.CABINET_RETREAT:
            n_steps = n_grasp_steps
            no_op = True
            grasp = False




        # Possibly override grasp value
        if grasp_override_val is not None:
            grasp = grasp_override_val

        grasp_val = -1.0 if grasp else 1.0
        null_cmds = None
        # If we're doing a no_op, don't move the EEF
        if no_op:
            # 움직이지 않고 고정된 pose에서만 grasp만 수행
            cmds = self.generate_no_ops(n_steps=n_steps, return_aa=True)        # 6개는 고정된 값 생성
            cmds = th.concatenate([cmds, th.ones((n_steps, 1)) * grasp_val], dim=-1)        # 마지막 1개는 open : 1, close : -1
        # else if delta joint vals is None, then we assume we want to converge to a static set point -- so we generate a
        # linearly-interpolated trajectory to the desired waypoint
        elif delta_jnt_vals is None:
            # 특정 위치로 이동만 할 때 
            if not self.target_step:
                target_pos, target_mat = self.compute_grasp_pose(
                    joint_to_grasp_pos=joint_to_grasp_pos,
                    delta_jnt_val=0.0,
                    return_mat=True,
                )
            else:
                _, target_mat = self.compute_grasp_pose(
                    joint_to_grasp_pos=joint_to_grasp_pos,
                    delta_jnt_val=0.0,
                    return_mat=True,
                )
                target_pos = joint_to_grasp_pos + th.tensor([0.0, 0.0, self.target_approach], dtype=th.float)
                

            target_pos_in_robot_frame, target_aa_in_robot_frame = \
                self.get_pose_in_robot_frame(pos=target_pos, mat=target_mat, return_mat=False, include_eef_offset=not maintain_current_orientation)

            # If we're in the approach stage, use that for planning instead!
            target_quat_in_robot_frame = OT.axisangle2quat(target_aa_in_robot_frame)

            # If maintaining current orientation, override the value!
            if maintain_current_orientation:
                target_quat_in_robot_frame = self._robot.get_relative_eef_orientation()

            # Compute commands
            cmds = self.interpolate_to_pose(
                target_pos=target_pos_in_robot_frame,
                target_quat=target_quat_in_robot_frame,
                n_steps=n_steps,
                return_aa=True,
            )
            # # print("cmds: ", cmds)
            # cmds = self.plan_to_pose(
            #     target_pos=target_pos_in_robot_frame,
            #     target_quat=target_quat_in_robot_frame,
            # )
            # # print("cmds: ", cmds)
            # exit()
            cmds = th.concatenate([cmds, th.ones((len(cmds), 1)) * grasp_val], dim=-1)

        # Otherwise, generate the trajectory directly from the joint vals requested
        else:
            # delta_jnt_vals : Cabinet과 같은 관절의 변화량
            # Grasp을 하면서 움직일 때
            cmds = th.zeros((n_steps, 7))
            for i, delta_jnt_val in enumerate(delta_jnt_vals):
                target_pos, target_mat = self.compute_grasp_pose(
                    joint_to_grasp_pos=joint_to_grasp_pos,
                    delta_jnt_val=delta_jnt_val,
                    return_mat=True,
                )
                target_pos_in_robot_frame, target_aa_in_robot_frame = \
                    self.get_pose_in_robot_frame(pos=target_pos, mat=target_mat, return_mat=False, include_eef_offset=not maintain_current_orientation)

                # If maintaining current orientation, override the value!
                if maintain_current_orientation:
                    target_aa_in_robot_frame = OT.quat2axisangle(self._robot.get_relative_eef_orientation())

                cmds[i] = th.concatenate([target_pos_in_robot_frame, target_aa_in_robot_frame, th.tensor([grasp_val], dtype=th.float)])    # pos, aa, grasp
            cmds = th.tensor(cmds, dtype=th.float)

        buffer_cmds = th.ones((n_buffer_steps, 7)) * cmds[-1].view(1, -1)
        cmds = th.concatenate([cmds, buffer_cmds], dim=0)

        if null_cmds is not None:
            buffer_null_cmds = th.ones((n_buffer_steps, null_cmds.shape[-1])) * null_cmds[-1].view(1, -1)
            null_cmds = th.concatenate([null_cmds, buffer_null_cmds], dim=0)

        # Possibly visualize
        if self._visualize:
            for marker_prim in self._progress_traj_markers.values():
                og.sim.scenes[0].remove_object(marker_prim)
            self._progress_traj_markers = dict()
            for i in range(len(cmds)):
                marker_name = f"marker_{i}"
                self._progress_traj_markers[marker_name] = PrimitiveObject(
                    name=f"open_close_skill_{self._target_obj.name}_{marker_name}",
                    primitive_type="Sphere",
                    visual_only=True,
                    radius=0.01,
                    rgba=[1.0, 0, 0, 1.0],
                )
                og.sim.scenes[0].add_object(self._progress_traj_markers[marker_name])

                if i == len(cmds) - 1:
                    last_act = cmds[-1]
                    target_pos, target_aa = last_act[:3], last_act[3:6]
                    target_orientation = OT.quat2mat(OT.axisangle2quat(target_aa))
                    self.visualize_marker(eef_pos=target_pos, eef_mat=target_orientation)

            self.visualize_traj_by_markers(cmds=cmds)

        return cmds, null_cmds

    def compute_gripper2handle_vector(self):
        # Get relative position of the grasping point in the joint frame
        joint_to_grasp_pos = self._joint_to_handle_pos

        # Convert that into global coordinates
        target_pos, target_mat = self.compute_grasp_pose(
            joint_to_grasp_pos=joint_to_grasp_pos,
            delta_jnt_val=0.0,
            return_mat=True,
        )

        # Convert that into the robot frame
        target_pos_in_robot_frame, target_aa_in_robot_frame = \
            self.get_pose_in_robot_frame(pos=target_pos, mat=target_mat, return_mat=False)

        # Get the robot end effector pose in the robot frame
        robot = self._robot
        robot_eef_pos, robot_eef_quat = robot.get_relative_eef_pose()

        # To get vector, subtract final - start
        gripper2handle_vector = target_pos_in_robot_frame - robot_eef_pos

        return gripper2handle_vector
    
    def visualize_traj_by_markers(self, cmds, pos_in_robot_frame=True):
        """
        Visualize a trajectory using markers
        Markers are defined in dictionary self._progress_traj_markers

        Args:
            cmds (torch.tensor): (T,D)-shaped tensor of commands
            pos_in_robot_frame (bool): whether the inputted @eef_pos and @eef_mat is specified in the robot frame or
                in global frame
        """
        for i, act in enumerate(cmds):
            marker_name = f"marker_{i}"
            cur_act = cmds[i]
            cur_target_pos, cur_target_aa = cur_act[:3], cur_act[3:6]
            cur_target_orientation = OT.quat2mat(OT.axisangle2quat(cur_target_aa))
            if pos_in_robot_frame:
                cur_target_pos, _ = self.get_pose_in_world_frame(pos=cur_target_pos, mat=cur_target_orientation, return_mat=False)
            self._progress_traj_markers[marker_name].set_position_orientation(position=cur_target_pos)

    def generate_no_ops(self, n_steps, return_aa=False):
        """
        Generates no-op actions to apply to the robot, returning EEF poses where the robot currently is located

        Args:
            n_steps (int): Number of no-op steps to generate
            return_aa (bool): Whether to return the orientations in quaternion or axis-angle representation

        Returns:
            torch.tensor: (n_steps, [6, 7])-shaped tensor where each entry is is the (x,y,z) position and (x,y,z,w)
                quaternion (if @return_aa is True) or (ax, ay, az) axis-angle orientation
        """
        cmds = th.zeros((n_steps, 6 if return_aa else 7))

        # Grab robot local pose
        cur_pos, cur_ori = self._robot.get_relative_eef_pose(mat=False)

        # Set as current command
        if return_aa:
            cur_ori = OT.quat2axisangle(cur_ori)

        cmds[:, :3] = cur_pos
        cmds[:, 3:] = cur_ori

        return cmds

    def reset_target_obj(self):
        """
        Resets the target object to its default state
        """
        self._target_obj.keep_still()
        # self._target_obj.set_joint_positions(th.zeros(self._target_obj.n_joints), drive=False)

    @property
    def steps(self):
        return OpenOrCloseStep
    
    @property
    def visualize_traj(self):
        return self._visualize

    @property
    def target_obj(self):
        return self._target_obj
