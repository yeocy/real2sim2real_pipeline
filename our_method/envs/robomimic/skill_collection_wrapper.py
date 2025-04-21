from our_method.envs.robomimic.data_collection_wrapper import DataCollectionWrapper
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
        # self.env.env : <our_method.envs.omnigibson.pick_cup_in_the_cabinet.PickCupInTheCabinetWrapper object at 0x7f2ec1e3eb30>

        # Iterate through all solve steps
        for solve_step in self.env.env.solve_steps:
            skill, skill_step, skill_kwargs, is_valid = self.env.env.get_skill_and_kwargs_at_step(solve_step=solve_step)
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

            # print(f"\n# --- Solve Step: {solve_step} ---")
            # print(f"# skill object: {skill}")
            # print(f"# skill class: {skill.__class__.__name__}")
            # print(f"# skill_step: {skill_step}")
            # print(f"# skill_kwargs: {skill_kwargs}")
            # print(f"# is_valid: {is_valid}")
            # print(f"# Executing solve_step: {solve_step} [skill {skill.__class__.__name__} step: {skill_step}]...")


            if not is_valid:
                print(f"Skill {skill.__class__.__name__} not valid at current sim state, terminating early")
                return

            print(f"Executing solve_step: {solve_step} [skill {skill.__class__.__name__} step: {skill_step}]...")
            actions, null_actions = skill.compute_current_subtrajectory(step=skill_step, **skill_kwargs)
            # === Skill Subtrajectory Outputs ===
            # Step: OpenOrCloseStep.APPROACH
            # Actions:
            # tensor([[-0.0068, -0.0215,  1.7870,  2.3652,  0.0322,  2.2718,  1.0000],
            #         [ 0.0059, -0.0153,  1.7866,  2.3015,  0.1495,  2.1994,  1.0000],
            #         [ 0.0186, -0.0091,  1.7862,  2.2345,  0.2596,  2.1245,  1.0000],
            #         [ 0.0313, -0.0029,  1.7858,  2.1646,  0.3627,  2.0475,  1.0000],
            #         [ 0.0440,  0.0033,  1.7854,  2.0920,  0.4591,  1.9686,  1.0000],
            #         [ 0.0566,  0.0095,  1.7849,  2.0171,  0.5493,  1.8880,  1.0000],
            #         [ 0.0693,  0.0157,  1.7845,  1.9401,  0.6335,  1.8059,  1.0000],
            #         [ 0.0820,  0.0218,  1.7841,  1.8612,  0.7120,  1.7226,  1.0000],
            #         [ 0.0947,  0.0280,  1.7837,  1.7807,  0.7851,  1.6382,  1.0000],
            #         [ 0.1074,  0.0342,  1.7833,  1.6986,  0.8529,  1.5529,  1.0000],
            #         [ 0.1201,  0.0404,  1.7829,  1.6152,  0.9156,  1.4667,  1.0000],
            #         [ 0.1328,  0.0466,  1.7824,  1.5306,  0.9736,  1.3799,  1.0000],
            #         [ 0.1455,  0.0528,  1.7820,  1.4449,  1.0269,  1.2924,  1.0000],
            #         [ 0.1582,  0.0590,  1.7816,  1.3583,  1.0758,  1.2044,  1.0000],
            #         [ 0.1709,  0.0652,  1.7812,  1.2707,  1.1203,  1.1160,  1.0000],
            #         [ 0.1709,  0.0652,  1.7812,  1.2707,  1.1203,  1.1160,  1.0000]])
            # Null Actions:
            # None
            # ===================================

            # === Skill Subtrajectory Outputs ===
            # Step: OpenOrCloseStep.CONVERGE
            # Actions:
            # tensor([[0.1742, 0.0619, 1.7809, 1.3196, 1.0958, 1.1642, 1.0000],
            #         [0.1836, 0.0610, 1.7809, 1.3161, 1.0976, 1.1608, 1.0000],
            #         [0.1929, 0.0600, 1.7810, 1.3126, 1.0994, 1.1573, 1.0000],
            #         [0.2022, 0.0590, 1.7810, 1.3091, 1.1011, 1.1539, 1.0000],
            #         [0.2115, 0.0581, 1.7810, 1.3056, 1.1029, 1.1504, 1.0000],
            #         [0.2209, 0.0571, 1.7810, 1.3022, 1.1047, 1.1470, 1.0000],
            #         [0.2302, 0.0562, 1.7810, 1.2987, 1.1064, 1.1436, 1.0000],
            #         [0.2395, 0.0552, 1.7810, 1.2952, 1.1082, 1.1401, 1.0000],
            #         [0.2489, 0.0543, 1.7810, 1.2917, 1.1099, 1.1367, 1.0000],
            #         [0.2582, 0.0533, 1.7811, 1.2882, 1.1117, 1.1332, 1.0000],
            #         [0.2675, 0.0523, 1.7811, 1.2847, 1.1134, 1.1298, 1.0000],
            #         [0.2768, 0.0514, 1.7811, 1.2812, 1.1151, 1.1263, 1.0000],
            #         [0.2862, 0.0504, 1.7811, 1.2777, 1.1169, 1.1229, 1.0000],
            #         [0.2955, 0.0495, 1.7811, 1.2742, 1.1186, 1.1194, 1.0000],
            #         [0.3048, 0.0485, 1.7811, 1.2707, 1.1203, 1.1160, 1.0000],
            #         [0.3048, 0.0485, 1.7811, 1.2707, 1.1203, 1.1160, 1.0000]])
            # Null Actions:
            # None
            # ===================================

            # === Skill Subtrajectory Outputs ===
            # Step: OpenOrCloseStep.GRASP
            # Actions:
            # tensor([[ 0.2985,  0.0492,  1.7810,  1.2731,  1.1191,  1.1183, -1.0000],
            #         [ 0.2985,  0.0492,  1.7810,  1.2731,  1.1191,  1.1183, -1.0000]])
            # Null Actions:
            # None
            # ===================================

            # === Skill Subtrajectory Outputs ===
            # Step: OpenOrCloseStep.ARTICULATE
            # Actions:
            # tensor([[ 0.3048,  0.0485,  1.7811,  1.2707,  1.1203,  1.1160, -1.0000],
            #         [ 0.3001,  0.0572,  1.7812,  1.2842,  1.0969,  1.0925, -1.0000],
            #         [ 0.2956,  0.0660,  1.7812,  1.2974,  1.0734,  1.0690, -1.0000],
            #         [ 0.2914,  0.0749,  1.7812,  1.3102,  1.0499,  1.0455, -1.0000],
            #         [ 0.2875,  0.0840,  1.7812,  1.3227,  1.0263,  1.0218, -1.0000],
            #         [ 0.2839,  0.0931,  1.7812,  1.3349,  1.0027,  0.9982, -1.0000],
            #         [ 0.2806,  0.1024,  1.7813,  1.3468,  0.9790,  0.9744, -1.0000],
            #         [ 0.2776,  0.1118,  1.7813,  1.3584,  0.9552,  0.9506, -1.0000],
            #         [ 0.2748,  0.1213,  1.7813,  1.3696,  0.9314,  0.9268, -1.0000],
            #         [ 0.2724,  0.1309,  1.7813,  1.3806,  0.9076,  0.9029, -1.0000],
            #         [ 0.2703,  0.1405,  1.7813,  1.3912,  0.8837,  0.8790, -1.0000],
            #         [ 0.2684,  0.1502,  1.7813,  1.4016,  0.8597,  0.8550, -1.0000],
            #         [ 0.2669,  0.1599,  1.7813,  1.4116,  0.8358,  0.8310, -1.0000],
            #         [ 0.2657,  0.1697,  1.7814,  1.4213,  0.8117,  0.8069, -1.0000],
            #         [ 0.2648,  0.1795,  1.7814,  1.4307,  0.7877,  0.7828, -1.0000],
            #         [ 0.2642,  0.1894,  1.7814,  1.4399,  0.7636,  0.7587, -1.0000],
            #         [ 0.2639,  0.1992,  1.7814,  1.4487,  0.7394,  0.7345, -1.0000],
            #         [ 0.2639,  0.2091,  1.7814,  1.4572,  0.7152,  0.7103, -1.0000],
            #         [ 0.2642,  0.2190,  1.7814,  1.4654,  0.6910,  0.6861, -1.0000],
            #         [ 0.2649,  0.2288,  1.7814,  1.4734,  0.6668,  0.6618, -1.0000],
            #         [ 0.2658,  0.2386,  1.7814,  1.4810,  0.6425,  0.6375, -1.0000],
            #         [ 0.2671,  0.2484,  1.7814,  1.4883,  0.6182,  0.6132, -1.0000],
            #         [ 0.2686,  0.2581,  1.7814,  1.4954,  0.5939,  0.5888, -1.0000],
            #         [ 0.2705,  0.2678,  1.7815,  1.5021,  0.5695,  0.5644, -1.0000],
            #         [ 0.2727,  0.2774,  1.7815,  1.5086,  0.5451,  0.5400, -1.0000],
            #         [ 0.2727,  0.2774,  1.7815,  1.5086,  0.5451,  0.5400, -1.0000]])
            # Null Actions:
            # None
            # ===================================

            # === Skill Subtrajectory Outputs ===
            # Step: OpenOrCloseStep.UNGRASP
            # Actions:
            # tensor([[0.2697, 0.2701, 1.7872, 1.4796, 0.5539, 0.5617, 1.0000],
            #         [0.2697, 0.2701, 1.7872, 1.4796, 0.5539, 0.5617, 1.0000]])
            # Null Actions:
            # None
            # ===================================



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


class SkillCollectionKinovaWrapper(DataCollectionWrapper):
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
        # self.env.env : <our_method.envs.omnigibson.pick_cup_in_the_cabinet.PickCupInTheCabinetWrapper object at 0x7f2ec1e3eb30>

        # Iterate through all solve steps
        for solve_step in self.env.env.solve_steps:
            skill, skill_step, skill_kwargs, is_valid = self.env.env.get_skill_and_kwargs_at_step(solve_step=solve_step)
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

            # print(f"\n# --- Solve Step: {solve_step} ---")
            # print(f"# skill object: {skill}")
            # print(f"# skill class: {skill.__class__.__name__}")
            # print(f"# skill_step: {skill_step}")
            # print(f"# skill_kwargs: {skill_kwargs}")
            # print(f"# is_valid: {is_valid}")
            # print(f"# Executing solve_step: {solve_step} [skill {skill.__class__.__name__} step: {skill_step}]...")


            if not is_valid:
                print(f"Skill {skill.__class__.__name__} not valid at current sim state, terminating early")
                return

            print(f"Executing solve_step: {solve_step} [skill {skill.__class__.__name__} step: {skill_step}]...")
            actions, null_actions = skill.compute_current_subtrajectory(step=skill_step, **skill_kwargs)
            # === Skill Subtrajectory Outputs ===
            # Step: OpenOrCloseStep.APPROACH
            # Actions:
            # tensor([[-0.0068, -0.0215,  1.7870,  2.3652,  0.0322,  2.2718,  1.0000],
            #         [ 0.0059, -0.0153,  1.7866,  2.3015,  0.1495,  2.1994,  1.0000],
            #         [ 0.0186, -0.0091,  1.7862,  2.2345,  0.2596,  2.1245,  1.0000],
            #         [ 0.0313, -0.0029,  1.7858,  2.1646,  0.3627,  2.0475,  1.0000],
            #         [ 0.0440,  0.0033,  1.7854,  2.0920,  0.4591,  1.9686,  1.0000],
            #         [ 0.0566,  0.0095,  1.7849,  2.0171,  0.5493,  1.8880,  1.0000],
            #         [ 0.0693,  0.0157,  1.7845,  1.9401,  0.6335,  1.8059,  1.0000],
            #         [ 0.0820,  0.0218,  1.7841,  1.8612,  0.7120,  1.7226,  1.0000],
            #         [ 0.0947,  0.0280,  1.7837,  1.7807,  0.7851,  1.6382,  1.0000],
            #         [ 0.1074,  0.0342,  1.7833,  1.6986,  0.8529,  1.5529,  1.0000],
            #         [ 0.1201,  0.0404,  1.7829,  1.6152,  0.9156,  1.4667,  1.0000],
            #         [ 0.1328,  0.0466,  1.7824,  1.5306,  0.9736,  1.3799,  1.0000],
            #         [ 0.1455,  0.0528,  1.7820,  1.4449,  1.0269,  1.2924,  1.0000],
            #         [ 0.1582,  0.0590,  1.7816,  1.3583,  1.0758,  1.2044,  1.0000],
            #         [ 0.1709,  0.0652,  1.7812,  1.2707,  1.1203,  1.1160,  1.0000],
            #         [ 0.1709,  0.0652,  1.7812,  1.2707,  1.1203,  1.1160,  1.0000]])
            # Null Actions:
            # None
            # ===================================

            # === Skill Subtrajectory Outputs ===
            # Step: OpenOrCloseStep.CONVERGE
            # Actions:
            # tensor([[0.1742, 0.0619, 1.7809, 1.3196, 1.0958, 1.1642, 1.0000],
            #         [0.1836, 0.0610, 1.7809, 1.3161, 1.0976, 1.1608, 1.0000],
            #         [0.1929, 0.0600, 1.7810, 1.3126, 1.0994, 1.1573, 1.0000],
            #         [0.2022, 0.0590, 1.7810, 1.3091, 1.1011, 1.1539, 1.0000],
            #         [0.2115, 0.0581, 1.7810, 1.3056, 1.1029, 1.1504, 1.0000],
            #         [0.2209, 0.0571, 1.7810, 1.3022, 1.1047, 1.1470, 1.0000],
            #         [0.2302, 0.0562, 1.7810, 1.2987, 1.1064, 1.1436, 1.0000],
            #         [0.2395, 0.0552, 1.7810, 1.2952, 1.1082, 1.1401, 1.0000],
            #         [0.2489, 0.0543, 1.7810, 1.2917, 1.1099, 1.1367, 1.0000],
            #         [0.2582, 0.0533, 1.7811, 1.2882, 1.1117, 1.1332, 1.0000],
            #         [0.2675, 0.0523, 1.7811, 1.2847, 1.1134, 1.1298, 1.0000],
            #         [0.2768, 0.0514, 1.7811, 1.2812, 1.1151, 1.1263, 1.0000],
            #         [0.2862, 0.0504, 1.7811, 1.2777, 1.1169, 1.1229, 1.0000],
            #         [0.2955, 0.0495, 1.7811, 1.2742, 1.1186, 1.1194, 1.0000],
            #         [0.3048, 0.0485, 1.7811, 1.2707, 1.1203, 1.1160, 1.0000],
            #         [0.3048, 0.0485, 1.7811, 1.2707, 1.1203, 1.1160, 1.0000]])
            # Null Actions:
            # None
            # ===================================

            # === Skill Subtrajectory Outputs ===
            # Step: OpenOrCloseStep.GRASP
            # Actions:
            # tensor([[ 0.2985,  0.0492,  1.7810,  1.2731,  1.1191,  1.1183, -1.0000],
            #         [ 0.2985,  0.0492,  1.7810,  1.2731,  1.1191,  1.1183, -1.0000]])
            # Null Actions:
            # None
            # ===================================

            # === Skill Subtrajectory Outputs ===
            # Step: OpenOrCloseStep.ARTICULATE
            # Actions:
            # tensor([[ 0.3048,  0.0485,  1.7811,  1.2707,  1.1203,  1.1160, -1.0000],
            #         [ 0.3001,  0.0572,  1.7812,  1.2842,  1.0969,  1.0925, -1.0000],
            #         [ 0.2956,  0.0660,  1.7812,  1.2974,  1.0734,  1.0690, -1.0000],
            #         [ 0.2914,  0.0749,  1.7812,  1.3102,  1.0499,  1.0455, -1.0000],
            #         [ 0.2875,  0.0840,  1.7812,  1.3227,  1.0263,  1.0218, -1.0000],
            #         [ 0.2839,  0.0931,  1.7812,  1.3349,  1.0027,  0.9982, -1.0000],
            #         [ 0.2806,  0.1024,  1.7813,  1.3468,  0.9790,  0.9744, -1.0000],
            #         [ 0.2776,  0.1118,  1.7813,  1.3584,  0.9552,  0.9506, -1.0000],
            #         [ 0.2748,  0.1213,  1.7813,  1.3696,  0.9314,  0.9268, -1.0000],
            #         [ 0.2724,  0.1309,  1.7813,  1.3806,  0.9076,  0.9029, -1.0000],
            #         [ 0.2703,  0.1405,  1.7813,  1.3912,  0.8837,  0.8790, -1.0000],
            #         [ 0.2684,  0.1502,  1.7813,  1.4016,  0.8597,  0.8550, -1.0000],
            #         [ 0.2669,  0.1599,  1.7813,  1.4116,  0.8358,  0.8310, -1.0000],
            #         [ 0.2657,  0.1697,  1.7814,  1.4213,  0.8117,  0.8069, -1.0000],
            #         [ 0.2648,  0.1795,  1.7814,  1.4307,  0.7877,  0.7828, -1.0000],
            #         [ 0.2642,  0.1894,  1.7814,  1.4399,  0.7636,  0.7587, -1.0000],
            #         [ 0.2639,  0.1992,  1.7814,  1.4487,  0.7394,  0.7345, -1.0000],
            #         [ 0.2639,  0.2091,  1.7814,  1.4572,  0.7152,  0.7103, -1.0000],
            #         [ 0.2642,  0.2190,  1.7814,  1.4654,  0.6910,  0.6861, -1.0000],
            #         [ 0.2649,  0.2288,  1.7814,  1.4734,  0.6668,  0.6618, -1.0000],
            #         [ 0.2658,  0.2386,  1.7814,  1.4810,  0.6425,  0.6375, -1.0000],
            #         [ 0.2671,  0.2484,  1.7814,  1.4883,  0.6182,  0.6132, -1.0000],
            #         [ 0.2686,  0.2581,  1.7814,  1.4954,  0.5939,  0.5888, -1.0000],
            #         [ 0.2705,  0.2678,  1.7815,  1.5021,  0.5695,  0.5644, -1.0000],
            #         [ 0.2727,  0.2774,  1.7815,  1.5086,  0.5451,  0.5400, -1.0000],
            #         [ 0.2727,  0.2774,  1.7815,  1.5086,  0.5451,  0.5400, -1.0000]])
            # Null Actions:
            # None
            # ===================================

            # === Skill Subtrajectory Outputs ===
            # Step: OpenOrCloseStep.UNGRASP
            # Actions:
            # tensor([[0.2697, 0.2701, 1.7872, 1.4796, 0.5539, 0.5617, 1.0000],
            #         [0.2697, 0.2701, 1.7872, 1.4796, 0.5539, 0.5617, 1.0000]])
            # Null Actions:
            # None
            # ===================================



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
