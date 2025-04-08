from digital_cousins.envs.robomimic.data_collection_wrapper import DataCollectionWrapper
import omnigibson.utils.transform_utils as OT
import torch as th
import h5py

h5py.get_config().track_order = True


class SkillCollectionWrapper(DataCollectionWrapper):
    """
    An OmniGibson environment wrapper for collecting skill-based data in robomimic format.
    """

    def __init__(self, env, path, only_successes=True, use_delta_commands=False):
        """
        Args:
            env (EB.EnvBase): The environment to wrap
            path (str): path to store robomimic hdf5 data file
            only_successes (bool): Whether to only save successful episodes
            use_delta_commands (bool): Whether robot should be using delta commands or not
        """
        self.use_delta_commands = use_delta_commands
        self._max_delta_action = None

        # Run super
        super().__init__(
            env=env,
            path=path,
            only_successes=only_successes,
        )

        # Make sure there's only one robot
        assert len(self.env.env.robots) == 1, f"Exactly one robot should exist in this {self.__class__.__name__}!"

        # If using delta commands, make sure input and output limits of robot arm controller are all +/- 1
        if self.use_delta_commands:
            arm_controller = self.env.env.robots[0].controllers[f"arm_{self.env.env.robots[0].default_arm}"]
            assert th.all(th.abs(arm_controller._command_input_limits[0]) == 1) and th.all(th.abs(arm_controller._command_input_limits[1]) == 1), \
                "All arm controller command input limits should be exactly +/-1.0!"
            # Compute the rescaling, first verify symmetric output limits
            assert th.all(th.abs(arm_controller._command_output_limits[0]) == th.abs(arm_controller._command_output_limits[1])), \
                "All arm controller command output limits should be symmetric!"
            self._max_delta_action = arm_controller._command_output_limits[1]

    def collect_demo(self):
        """
        Collects a single demonstration of @skill using @robot with any necessary arguments for the skill
        """
        # Execute the skill
        robot = self.env.env.robots[0]
        arm_name = f"arm_{robot.default_arm}"
        eef_name = f"gripper_{robot.default_arm}"

        # === 로봇 정보 ===
        # robot: <omnigibson.robots.franka_mounted.FrankaMounted object at 0x7fb502c4cc10>
        # arm_name: arm_0
        # eef_name: gripper_0
        # robot action_dim: 7
        # default_arm: 0
        # controller_action_idx: {'arm_0': tensor([0, 1, 2, 3, 4, 5]), 'gripper_0': tensor([6])}

        # Iterate through all solve steps
        for solve_step in self.env.env.solve_steps:
            print(f"\n--- Solve Step: {solve_step} ---")

            skill, skill_step, skill_kwargs, is_valid = self.env.env.get_skill_and_kwargs_at_step(solve_step=solve_step)

            print(f"skill object: {skill}")
            print(f"skill class: {skill.__class__.__name__}")
            print(f"skill_step: {skill_step}")
            print(f"skill_kwargs: {skill_kwargs}")
            print(f"is_valid: {is_valid}")

            # --- Solve Step: 0 ---
            # skill object: <digital_cousins.skills.open_or_close_skill.OpenOrCloseSkill object at 0x7fb5452258a0>
            # skill class: OpenOrCloseSkill
            # skill_step: 0
            # skill_kwargs: {'should_open': True, 'joint_limits': (0.0, 0.7853981633974483), 'n_approach_steps': 15, 'n_converge_steps': 15, 'n_grasp_steps': 1, 'n_articulate_steps': 25, 'n_buffer_steps': 1}
            # is_valid: True
            # Executing solve_step: 0 [skill OpenOrCloseSkill step: 0]...
            # UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).

            # --- Solve Step: 1 ---
            # skill object: <digital_cousins.skills.open_or_close_skill.OpenOrCloseSkill object at 0x7f0831254670>
            # skill class: OpenOrCloseSkill
            # skill_step: 1
            # skill_kwargs: {'should_open': True, 'joint_limits': (0.0, 0.7853981633974483), 'n_approach_steps': 15, 'n_converge_steps': 15, 'n_grasp_steps': 1, 'n_articulate_steps': 25, 'n_buffer_steps': 1}
            # is_valid: True
            # Executing solve_step: 1 [skill OpenOrCloseSkill step: 1]...

            # --- Solve Step: 2 ---
            # skill object: <digital_cousins.skills.open_or_close_skill.OpenOrCloseSkill object at 0x7f0831254670>
            # skill class: OpenOrCloseSkill
            # skill_step: 2
            # skill_kwargs: {'should_open': True, 'joint_limits': (0.0, 0.7853981633974483), 'n_approach_steps': 15, 'n_converge_steps': 15, 'n_grasp_steps': 1, 'n_articulate_steps': 25, 'n_buffer_steps': 1}
            # is_valid: True
            # Executing solve_step: 2 [skill OpenOrCloseSkill step: 2]...

            # --- Solve Step: 3 ---
            # skill object: <digital_cousins.skills.open_or_close_skill.OpenOrCloseSkill object at 0x7f0831254670>
            # skill class: OpenOrCloseSkill
            # skill_step: 3
            # skill_kwargs: {'should_open': True, 'joint_limits': (0.0, 0.7853981633974483), 'n_approach_steps': 15, 'n_converge_steps': 15, 'n_grasp_steps': 1, 'n_articulate_steps': 25, 'n_buffer_steps': 1}
            # is_valid: True
            # Executing solve_step: 3 [skill OpenOrCloseSkill step: 3]...
            # UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).

            # --- Solve Step: 4 ---
            # skill object: <digital_cousins.skills.open_or_close_skill.OpenOrCloseSkill object at 0x7f0831254670>
            # skill class: OpenOrCloseSkill
            # skill_step: 4
            # skill_kwargs: {'should_open': True, 'joint_limits': (0.0, 0.7853981633974483), 'n_approach_steps': 15, 'n_converge_steps': 15, 'n_grasp_steps': 1, 'n_articulate_steps': 25, 'n_buffer_steps': 1}
            # is_valid: True
            # Executing solve_step: 4 [skill OpenOrCloseSkill step: 4]...


            if not is_valid:
                print(f"Skill {skill.__class__.__name__} not valid at current sim state, terminating early")
                return

            print(f"Executing solve_step: {solve_step} [skill {skill.__class__.__name__} step: {skill_step}]...")
            actions, null_actions = skill.compute_current_subtrajectory(step=skill_step, **skill_kwargs)

            if actions is None:
                print(f"Failed skill {skill_step}, terminating early")
                return

            for i, act in enumerate(actions):
                action = th.zeros(robot.action_dim)
                if self.use_delta_commands:
                    robot_pos, robot_aa = act[:3], act[3:6]
                    curr_eef_pos = robot.get_relative_eef_position()
                    curr_eef_quat = robot.get_relative_eef_orientation()
                    delta_pos = robot_pos - curr_eef_pos
                    delta_aa = OT.quat2axisangle(OT.quat_distance(OT.axisangle2quat(robot_aa), curr_eef_quat))
                    arm_act = th.concatenate([delta_pos, delta_aa]).clip(min=-self._max_delta_action, max=self._max_delta_action) / self._max_delta_action
                else:
                    arm_act = act[:-1]
                action[robot.controller_action_idx[arm_name]] = arm_act
                action[robot.controller_action_idx[eef_name]] = act[-1]

                # Update null space for arm OSC if null action is specified
                if null_actions is not None:
                    robot.controllers[f"arm_{robot.default_arm}"].default_joint_pos = null_actions[i]

                obs, r, terminated, truncated, info = self.step(action=action)
