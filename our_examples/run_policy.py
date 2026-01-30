import yaml
from manipulation.utils import save_numpy_as_gif, build_up_env
from RL.ray_learn import load_policy, make_env  
import numpy as np
import os
import ray
from ray import tune

if not ray.is_initialized():
    ray.init(num_cpus=1, ignore_reinit_error=True, log_to_driver=False)

# task_config_path = "/home/kodogyu/github_repos/RoboGen/data/generated_task_from_ours/Pick_up_the_water_bottle_on_the_desk_water_bottle__2025-04-23-12-45-41/Pick_up_the_water_bottle_on_the_desk_The_robot_arm_needs_to_approach_the_water_bottle_on_the_desk_grasp_it_securely_and_then_lift_it_up_from_the_desk.yaml"
# task_name = "lift_the_water_bottle_from_the_desk"
# last_restore_state_file = "/home/kodogyu/github_repos/RoboGen/data/generated_task_from_ours/Pick_up_the_water_bottle_on_the_desk_water_bottle__2025-04-23-12-45-41/task_Pick_up_the_water_bottle_on_the_desk/primitive_states/2025-04-23-17-23-36/grasp_the_water_bottle/state_141.pkl"
# load_policy_path = "/home/kodogyu/github_repos/RoboGen/data/generated_task_from_ours/Pick_up_the_water_bottle_on_the_desk_water_bottle__2025-04-23-12-45-41/task_Pick_up_the_water_bottle_on_the_desk/RL_sac/2025-04-24-00-01-23/lift_the_water_bottle_from_the_desk/best_model/checkpoint_000399/checkpoint-399"
# load_policy_path = "/home/kodogyu/github_repos/RoboGen/data/generated_task_from_ours/Pick_up_the_water_bottle_on_the_desk_water_bottle__2025-04-23-12-45-41/task_Pick_up_the_water_bottle_on_the_desk/RL_sac/2025-04-25-11-28-33/lift_the_water_bottle_from_the_desk/best_model/checkpoint_000849/checkpoint-849"
# load_policy_path = "/home/kodogyu/github_repos/RoboGen/data/generated_task_from_description/Pick_up_the_water_bottle_on_the_desk_Bottle_4064_2025-04-18-18-10-06/task_Pick_up_the_water_bottle_on_the_desk/RL_sac/2025-04-25-11-21-42/lift_the_bottle_from_the_desk/best_model/checkpoint_001149/checkpoint-1149"
# load_policy_path = "/home/kodogyu/github_repos/RoboGen/data/generated_task_from_ours/Put_the_apple_in_the_basket/task_Put_the_apple_in_the_basket/RL_sac/2025-04-26-02-26-45/move_the_apple_over_the_basket/best_model/checkpoint_002299/checkpoint-2299"
# load_policy_path = "/home/kodogyu/github_repos/RoboGen/data/generated_task_from_description/Put_the_apple_in_the_basket_Basket_100466_2025-04-25-16-46-01/task_Put_the_apple_in_the_basket/RL_sac/2025-04-25-21-34-13/move_the_apple_over_the_basket/best_model/checkpoint_002400/checkpoint-2400"
# load_policy_path = "/home/kodogyu/github_repos/RoboGen/data/generated_task_from_ours/Heat_up_the_kettle/task_Heat_up_the_kettle/RL_sac/2025-04-27-02-42-17/move_the_kettle_to_the_stove/best_model/checkpoint_002499/checkpoint-2499"
# load_policy_path = "/home/kodogyu/github_repos/RoboGen/data/generated_task_from_description/Heat_up_the_kettle_Kettle_100031_2025-04-27-01-08-50/task_Heat_up_the_kettle/RL_sac/2025-04-27-17-09-04/move_the_kettle_to_the_stove/best_model/checkpoint_001249/checkpoint-1249"
load_policy_path = "/home/kodogyu/github_repos/RoboGen/data/generated_task_from_ours/Take_out_the_sauce_pot/task_Take_out_the_sauce_pot/RL_sac/2025-04-29-21-43-56/lift_the_pot/best_model/checkpoint_001849/checkpoint-1849"

# task_config_path = "/home/kodogyu/github_repos/RoboGen/data/generated_from_experiments/exp1_2_test1/robogen_exp_config.yaml"
# task_config_path = "/home/kodogyu/github_repos/RoboGen/data/generated_task_from_ours/Pick_up_the_water_bottle_on_the_desk_water_bottle__2025-04-23-12-45-41/Pick_up_the_water_bottle_on_the_desk_The_robot_arm_needs_to_approach_the_water_bottle_on_the_desk_grasp_it_securely_and_then_lift_it_up_from_the_desk.yaml"
# task_config_path = "data/generated_task_from_ours/Put_the_apple_in_the_basket/robogen_ours_config.yaml"
# task_config_path = "/home/kodogyu/github_repos/RoboGen/data/generated_from_experiments/exp3_a_test/test_scene_obs_target_only/test_scene_configs1/robogen_exp_config1_scene_0.yaml"
# task_config_path = "/home/kodogyu/github_repos/RoboGen/data/generated_from_experiments/exp3_a_test/test_scene_obs_target_only/test_scene_configs2/robogen_exp_config2_scene_0.yaml"
# task_config_path = "/home/kodogyu/github_repos/RoboGen/data/generated_from_experiments/exp4_test/exp4_test1/test_scene_configs1/robogen_exp_config1_scene_0.yaml"
task_config_path = "/home/kodogyu/github_repos/RoboGen/data/generated_from_experiments/exp4_test/exp4_test1/test_scene_configs1/robogen_exp_config1_scene_1.yaml"
# task_config_path = "data/generated_from_experiments/exp2_test/robogen_exp_config1.yaml"
# task_name = "lift_the_water_bottle_from_the_desk"
# task_name = "move_the_apple_over_the_basket"
# task_name = "move_the_kettle_to_the_stove"
task_name = "lift_the_pot"
# last_restore_state_file = "/home/kodogyu/github_repos/RoboGen/data/generated_from_experiments/exp1_2_test1/task_Experiment_Task/primitive_states/config1/2025-04-24-22-30-30_scene_0/grasp_the_water_bottle/state_141.pkl"
# last_restore_state_file = "/home/kodogyu/github_repos/RoboGen/data/generated_task_from_ours/Pick_up_the_water_bottle_on_the_desk_water_bottle__2025-04-23-12-45-41/task_Pick_up_the_water_bottle_on_the_desk/primitive_states/2025-04-23-17-23-36/grasp_the_water_bottle/state_141.pkl"
# last_restore_state_file = "data/generated_task_from_ours/Put_the_apple_in_the_basket/task_Put_the_apple_in_the_basket/primitive_states/2025-04-26-02-04-57/grasp_the_apple/state_134.pkl"
# last_restore_state_file = "/home/kodogyu/github_repos/RoboGen/data/generated_from_experiments/exp2_test/task_Experiment_Task/primitive_states/config6/2025-04-26-19-18-58_scene_0/grasp_the_apple/state_149.pkl"
# last_restore_state_file = "/home/kodogyu/github_repos/RoboGen/data/generated_from_experiments/exp3_a_test/task_Experiment_Task/primitive_states/config1/2025-04-27-17-35-25_scene_0/grasp_the_kettle_body/state_149.pkl"
# last_restore_state_file = "/home/kodogyu/github_repos/RoboGen/data/generated_from_experiments/exp3_a_test/task_Experiment_Task/primitive_states/config2/2025-04-27-17-42-09_scene_0/grasp_the_kettle_body/state_149.pkl"
# last_restore_state_file = "/home/kodogyu/github_repos/RoboGen/data/generated_from_experiments/exp4_test/exp4_test1/task_Experiment_Task/primitive_states/config1/2025-04-30-01-49-10_scene_0/open_the_cabinet_and_grasp_the_pot1/state_130.pkl"
last_restore_state_file = "/home/kodogyu/github_repos/RoboGen/data/generated_from_experiments/exp4_test/exp4_test1/task_Experiment_Task/primitive_states/config1/2025-04-30-01-49-10_scene_1/open_the_cabinet_and_grasp_the_pot1/state_149.pkl"

observe_only_target_obj = False  #! 주의
use_gpt_spatial_relationship = False
# action_space = "normalized-direct-translation"  #! 주의
action_space = "delta-translation"  #! 주의
# build the environment
# NOTE: change to your taks config path
# task_config_path = "data/generated_task_from_description/open_the_storage_furniture_StorageFurniture_48452_2023-12-25-16-50-52/open_the_storage_furniture_The_robot_arm_will_open_the_storage_furniture_such_as_a_cabinet_or_a_drawer.yaml"
# task_config_path = "data/generated_task_from_description/MY_INSTRUCTION/MY_INSTRUCTION.yaml"
# task_config_path = "data/generated_task_from_ours/Pick_up_the_water_bottle_on_the_table_water_bottle__2025-04-19-23-08-03/Pick_up_the_water_bottle_on_the_table_The_robot_arm_needs_to_approach_the_water_bottle_on_the_table_grasp_it_securely_and_then_lift_it_up_from_the_table.yaml"

with open(task_config_path, 'r') as file:
    task_config = yaml.safe_load(file)

solution_path = None
for obj in task_config:
    if "solution_path" in obj:
        solution_path = obj["solution_path"]
        break

# NOTE: change to your task name
# task_name = "open_the_door_of_the_storage_furniture"
# task_name = "lift_the_water_bottle_from_the_table"

# NOTE: this is important, this should be set to final state before running the RL algorithm. Change to your state file
# last_restore_state_file = "data/generated_task_from_description/open_the_storage_furniture_StorageFurniture_48452_2023-12-25-16-50-52/task_open_the_storage_furniture/primitive_states/2023-12-25-17-05-33/grasp_the_door_of_the_storage_furniture/state_134.pkl" 
# last_restore_state_file = "data/generated_task_from_description/MY_INSTRUCTION/task_MY_INSTRUCTION/primitive_states/2025-04-19-01-07-13/grasp_the_body_of_the_bottle/state_137.pkl" 
# last_restore_state_file = "data/generated_task_from_ours/Pick_up_the_water_bottle_on_the_table_water_bottle__2025-04-19-23-08-03/task_Pick_up_the_water_bottle_on_the_table/primitive_states/2025-04-19-23-45-09/grasp_the_water_bottle/state_135.pkl" 

obj_id = 0
gui = True
randomize = False
robot_scale = 1

env_config = {
    "task_config_path": task_config_path,
    "solution_path": solution_path,
    "task_name": task_name,
    "last_restore_state_file": last_restore_state_file,
    "action_space": action_space, # NOTE: use the proper action space for the task
    "randomize": randomize,
    "use_bard": True,
    "obj_id": obj_id,
    "table_id": 0,
    "camera_position": [-1.018,-1.3939,1.7417],
    # "camera_orientation": [0.4685,-0.0067,0.0,0.8834],
    "robot_name": 'panda',
    "robot_scale": robot_scale,
    "observe_only_target_obj": observe_only_target_obj,
    "use_gpt_size": True,
    "use_gpt_joint_angle": True,
    "use_gpt_spatial_relationship": use_gpt_spatial_relationship,
    "use_distractor": True,
    'reward_trajectory_save_path': ""
}

env = make_env(env_config, render=gui)

env_name = task_name
tune.register_env(env_name, lambda config: make_env(config))


# load the policy
algo = 'sac'
# NOTE: change to your policy path
# load_policy_path = "data/generated_task_from_description/open_the_storage_furniture_StorageFurniture_48452_2023-12-25-16-50-52/task_open_the_storage_furniture/RL_sac/2023-12-25-17-05-33/open_the_door_of_the_storage_furniture/best_model/checkpoint_001349/checkpoint-1349"
# load_policy_path = "data/generated_task_from_description/MY_INSTRUCTION/task_MY_INSTRUCTION/RL_sac/2025-04-19-01-27-25/lift_the_bottle_from_the_desk/best_model/checkpoint_002249/checkpoint-2249"
# load_policy_path = "data/generated_task_from_ours/Pick_up_the_water_bottle_on_the_table_water_bottle__2025-04-19-23-08-03/task_Pick_up_the_water_bottle_on_the_table/RL_sac/2025-04-19-23-54-50/lift_the_water_bottle_from_the_table/best_model/checkpoint_002199/checkpoint-2199"
# load_policy_path = "data/generated_task_from_ours/Pick_up_the_water_bottle_on_the_table_water_bottle__2025-04-20-19-13-08/task_Pick_up_the_water_bottle_on_the_table/RL_sac/2025-04-20-20-32-37/lift_the_water_bottle_from_the_table/best_model/checkpoint_001199/checkpoint-1199"

agent, checkpoint_path = load_policy(algo, env_name, load_policy_path, env_config=env_config, seed=0)

obs = env.reset()
done = False
ret = 0
rgbs = []
state_files = []
states = []
while not done:
    # Compute the next action using the trained policy
    action = agent.compute_action(obs, explore=False)
    print("action: ", action)
    # Step the simulation forward using the action from our trained policy
    obs, reward, done, info = env.step(action)
    ret += reward
    rgb, depth = env.render()
    rgbs.append(rgb)
        
save_numpy_as_gif(np.array(rgbs), "data/eval.gif")

