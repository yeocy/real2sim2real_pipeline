import omnigibson as og
from omnigibson.robots import ManipulationRobot
import omnigibson.utils.transform_utils as OT
import torch as th


DEFAULT_VISUALIZE_CAM_POSE = (
    th.tensor([-0.36549226, 0.00287964, 1.33848734]),
    th.tensor([-0.30662805, 0.31306809, 0.64211894, -0.62900785]),
)


class SyntheticSkill:
    """
    Very basic class for implementing and deploying synthetic skills in OG. "Synthetic" implies access to ground truth,
    privileged information
    """
    def __init__(
        self,
        robot,
        visualize=False,
        visualize_cam_pose=None,
    ):
        """
        Args:
            robot (BaseRobot): Robot on which to deploy the skill
            visualize (bool): Whether to visualize this skill or not
            visualize_cam_pose (None or 2-tuple of torch.tensor): If specified, the relative pose to place the
                viewer camera wrt to the robot's root link. Otherwise, will use a hardcoded default
        """
        # Make sure sim is playing
        assert og.sim.is_playing(), "Simulator must be playing in order to initialize a SyntheticSkill!"

        # Store internal information
        self._robot = robot
        self._scene = robot.scene
        self._visualize = visualize
        self._visualize_cam_pose = DEFAULT_VISUALIZE_CAM_POSE if visualize_cam_pose is None else (th.Tensor(visualize_cam_pose[0]), th.Tensor(visualize_cam_pose[1]))

        # Create variables that will be populated later
        self._marker = None

        # Initialize this skill
        self.initialize()

    def initialize(self):
        """
        Runs any initialization necessary for this skill. Default is no-op
        """
        pass

    def set_camera_to_visualize(self):
        """
        Places global sim viewer camera at a hardcoded relative pose wrt the robot to better visualize the skill
        """
        # Cam relative pose wrt the robot
        cam_pos_in_robot_frame, cam_quat_in_robot_frame = self._visualize_cam_pose

        # Transfer to global pose
        robot_pos, robot_quat = self._robot.get_position_orientation()
        robot_mat = OT.quat2mat(robot_quat)
        cam_pos_global = robot_pos + robot_mat @ cam_pos_in_robot_frame
        cam_quat_global = OT.mat2quat(robot_mat @ OT.quat2mat(cam_quat_in_robot_frame))

        og.sim.viewer_camera.set_position_orientation(cam_pos_global, cam_quat_global)

        # Render a bit
        for i in range(5):
            og.sim.render()

    def compute_current_subtrajectory(self, *args, **kwargs):
        """
        Computes the desired subtrajectory at the current step in executing the skill. Takes in arbitrary arguments.
        Should be implemented by subclass

        Returns:
            2-tuple:
                - th.tensor: (T, D)-shaped array where D-length actions are stacked to form an T-length
                    subtrajectory action sequence to deploy in an environment
                - dict: Any relevant info for the generated subtrajectory
        """
        raise NotImplementedError()

    def reset(self):
        """
        Run any necessary resets for this skill. Should occur every time the environment is reset.
        """
        # no-op by default
        pass

    @property
    def steps(self):
        """
        Returns:
            IntEnum class: Skill steps for this skill
        """
        raise NotImplementedError()


class ManipulationSkill(SyntheticSkill):
    """
    Skill for Manipulation robots
    """
    def __init__(
        self,
        robot,
        eef_z_offset=0.093,
        visualize=False,
        visualize_cam_pose=None,
    ):
        """
        Args:
            robot (BaseRobot): Robot on which to deploy the skill
            eef_z_offset (float): Distance in the robot's EEF z-direction specifying distance to its actual grasping
                location for its assumed parallel jaw gripper
            visualize (bool): Whether to visualize this skill or not
            visualize_cam_pose (None or 2-tuple of torch.tensor): If specified, the relative pose to place the
                viewer camera wrt to the robot's root link. Otherwise, will use a hardcoded default
        """
        # Make sure the robot is a manipulator robot
        assert isinstance(robot, ManipulationRobot), f"ManipulationRobot needed for {self.__class__.__name__}!"

        # Store internal information
        self._eef_z_offset = eef_z_offset

        # Run super
        super().__init__(
            robot=robot,
            visualize=visualize,
            visualize_cam_pose=visualize_cam_pose,
        )

    def visualize_marker(self, eef_pos, eef_mat, pos_in_robot_frame=True):
        """
        Visualize the marker at the corresponding @eef_pos

        Args:
            eef_pos (torch.tensor): (x,y,z) end effector position
            eef_mat (torch.tensor): (3,3)-shaped end effector orientation matrix
            pos_in_robot_frame (bool): whether the inputted @eef_pos and @eef_mat is specified in the robot frame or
                in global frame
        """
        # Convert to world frame if needed
        if pos_in_robot_frame:
            eef_pos, _ = self.get_pose_in_world_frame(pos=eef_pos, mat=eef_mat, return_mat=False)

        self._marker.set_position_orientation(position=eef_pos)

    def get_pose_in_robot_frame(self, pos, mat, return_mat=False, include_eef_offset=True):
        """
        Converts a global pose from @pos and @mat into the equivalent pose expressed in @self._robot's frame

        Args:
            pos (torch.tensor): (x,y,z) global position to convert into robot frame
            mat (torch.tensor): (3,3)-shaped global orientation matrix to convert into robot frame
            return_mat (bool): Whether to return the orientation as a 3x3 matrix or a 3-array axis-angle representation
            include_eef_offset (bool): Whether to take into account the robot's eef offset when computing
                the corresponding robot pose

        Returns:
            2-tuple:
                - torch.tensor: (x,y,z) position in the robot frame
                - torch.tensor: (ax,ay,az) axis-angle orientation or (3,3)-shaped orientation matrix in the robot frame
        """
        # Get desired grasp pose (marker pos, facing cabinet) in robot frame
        robot_pos, robot_quat = self._robot.get_position_orientation()
        robot_mat = OT.quat2mat(robot_quat)
        pos_in_robot_frame = robot_mat.T @ (pos - robot_pos)
        mat_in_robot_frame = robot_mat.T @ mat
        aa_in_robot_frame = OT.quat2axisangle(OT.mat2quat(mat_in_robot_frame))

        # Modify grasp pose to offset for EEF-to-gripper frame
        eef_link_to_grasp_pos = th.tensor([0, 0, self._eef_z_offset if include_eef_offset else 0.0], dtype=th.float)      # z-dist from EEF frame to location of actual point of contact for the parallel jaw gripper
        pos_in_robot_frame -= mat_in_robot_frame @ eef_link_to_grasp_pos

        return pos_in_robot_frame, (mat_in_robot_frame if return_mat else aa_in_robot_frame)

    def get_pose_in_world_frame(self, pos, mat, return_mat=False):
        """
        Converts a pose from @pos and @mat expressed in @self._robot's frame into the equivalent pose expressed
        in the world frame

        NOTE: Inverts the end-effector offset as well

        Args:
            pos (torch.tensor): (x,y,z) local position to convert from robot frame into world frame
            mat (torch.tensor): (3,3)-shaped local orientation matrix to convert from robot frame into world frame
            return_mat (bool): Whether to return the orientation as a 3x3 matrix or a 3-array axis-angle representation

        Returns:
            2-tuple:
                - torch.tensor: (x,y,z) position in the local robot frame
                - torch.tensor: (ax,ay,az) axis-angle orientation or (3,3)-shaped orientation matrix in the global frame
        """
        # Get desired grasp pose (marker pos, facing cabinet) in robot frame
        robot_pos, robot_quat = self._robot.get_position_orientation()
        robot_mat = OT.quat2mat(robot_quat)
        mat_in_world_frame = robot_mat @ mat

        # Modify grasp pose to offset for EEF-to-gripper frame
        eef_link_to_grasp_pos = th.tensor([0, 0, self._eef_z_offset], dtype=th.float)      # z-dist from EEF frame to location of actual point of contact for the parallel jaw gripper
        pos_in_world_frame = robot_mat @ pos + mat_in_world_frame @ eef_link_to_grasp_pos + robot_pos
        aa_in_world_frame = OT.quat2axisangle(OT.mat2quat(mat_in_world_frame))

        return pos_in_world_frame, (mat_in_world_frame if return_mat else aa_in_world_frame)

    def interpolate_to_pose(self, target_pos, target_quat, n_steps, return_aa=False):
        """
        Generates interpolated set of poses bridging the robot's current EEF pose to the desired @target_pos and
        @target_quat specified in the robot's local frame

        Args:
            target_pos (torch.tensor): (x,y,z) desired final local EEF position specified in robot frame
            target_quat (torch.tensor): (x,y,z,w) desired final local EEF quaternion specified in robot frame
            n_steps (int): Number of total steps defining the interpolated trajectory
            return_aa (bool): Whether to return the interpolated orientations in quaternion or axis-angle representation

        Returns:
            torch.tensor: (n_steps, [6, 7])-shaped array where each entry is is the (x,y,z) position and (x,y,z,w)
                quaternion (if @return_aa is False) or (ax, ay, az) axis-angle orientation
        """
        # Grab robot local pose
        cur_pos, cur_quat = self._robot.get_relative_eef_pose(mat=False)

        # Get from pose0 --> pose1 in n_steps
        # Returns list of 2-tuple, where each entry is an interpolated value between the endpoints
        # Note: Does NOT include the start point
        vals = th.zeros((n_steps, 6 if return_aa else 7))
        pos_delta = target_pos - cur_pos
        for i in range(n_steps):
            frac = th.tensor((i + 1) / n_steps)
            ori = OT.quat_slerp(cur_quat, target_quat, frac=frac)
            if return_aa:
                ori = OT.quat2axisangle(ori)
            vals[i, :3] = cur_pos + pos_delta * frac
            vals[i, 3:] = ori

        return th.tensor(vals, dtype=th.float)

    # def plan_to_pose(self, target_pos, target_quat, n_steps=50):
    #     """
    #     Plans a joint trajectory to a target EEF pose using interpolation + IK (no planning context, no collision check)

    #     Args:
    #         target_pos (torch.tensor): (x,y,z) EEF target position (in robot frame)
    #         target_quat (torch.tensor): (x,y,z,w) EEF target quaternion (in robot frame)
    #         n_steps (int): Number of steps in the interpolated path

    #     Returns:
    #         List[torch.Tensor] or None: List of joint positions along the trajectory, or None if IK fails
    #     """
    #     from omnigibson.utils.control_utils import IKSolver
    #     # Setup joint info
    #     arm_name = self._robot.default_arm
    #     joint_control_idx = self._robot.arm_control_idx[arm_name]
    #     current_joint_pos = self._robot.get_joint_positions()[joint_control_idx]

    #     # IK solver 준비
    #     ik_solver = IKSolver(
    #         robot_description_path=self._robot.robot_arm_descriptor_yamls.get("left_fixed", self._robot.robot_arm_descriptor_yamls[arm_name]),
    #         robot_urdf_path=self._robot.urdf_path,
    #         reset_joint_pos=self._robot.reset_joint_pos[joint_control_idx],
    #         eef_name=self._robot.eef_link_names[arm_name],
    #     )

    #     # 현재 EEF pose
    #     cur_pos, cur_quat = self._robot.get_relative_eef_pose(mat=False)

    #     # trajectory 생성
    #     traj = []
    #     for i in range(1, n_steps + 1):
    #         frac = th.tensor(i / n_steps, dtype=th.float32)
    #         interp_pos = cur_pos * (1 - frac) + target_pos * frac
    #         interp_quat = OT.quat_slerp(cur_quat, target_quat, frac=frac)

    #         # IK로 joint 값 구하기
    #         joint_sol = ik_solver.solve(
    #             target_pos=interp_pos,
    #             target_quat=interp_quat,
    #             max_iterations=1000,
    #         )

    #         if joint_sol is not None:
    #             traj.append(joint_sol)
    #         else:
    #             print(f"[⚠️ IK 실패] Step {i}/{n_steps}: pos={interp_pos}, quat={interp_quat}")
    #             return None  # 중간에 실패하면 전체 실패로 간주
    #     # if len(traj) == n_steps:
    #     #     return th.stack(traj, dim=0)
    #     # else:
    #     #     return None
    #     if len(traj) == n_steps:
    #         eef_traj = []
    #         for joint_pos in traj:
    #             self._robot.set_joint_positions(joint_pos, indices=joint_control_idx)
    #             pos, quat = self._robot.get_relative_eef_pose(mat=False)
    #             aa = OT.quat2axisangle(quat)
    #             pose = th.cat([pos, aa])
    #             eef_traj.append(pose)
    #         return th.stack(eef_traj, dim=0)
    #     else:
    #         return None