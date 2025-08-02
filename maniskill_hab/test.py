import gymnasium as gym

from mani_skill import ASSET_DIR
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
import mshab.envs
from mshab.envs.planner import plan_data_from_file
import matplotlib.pyplot as plt

task = "set_table" # "tidy_house", "prepare_groceries", or "set_table"
subtask = "open"    # "sequential", "pick", "place", "open", "close"
                    # NOTE: sequential loads the full task, e.g pick -> place -> ...
                    #     while pick, place, etc only simulate a single subtask each episode
split = "train"     # "train", "val"


REARRANGE_DIR = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange"

plan_data = plan_data_from_file(
    REARRANGE_DIR / "task_plans" / task / subtask / split / "fridge_new.json"
)
spawn_data_fp = REARRANGE_DIR / "spawn_data" / task / subtask / split / "spawn_data.pt"

import torch
spawn_data= torch.load(spawn_data_fp)
# print(spawn_data)

env = gym.make(
    f"{subtask.capitalize()}SubtaskTrain-v0",
    # Simulation args
    num_envs=1,  # RCAD has 63 train scenes, so 252 envs -> 4 parallel envs reserved for each scene
    obs_mode="state", #"rgbd",
    sim_backend="gpu",
    # robot_uids="fetch",
    robot_uids="panda",
    control_mode="pd_joint_delta_pos",
    # Rendering args
    reward_mode="normalized_dense",
    render_mode= "human", #"rgb_array",
    # shader_dir="minimal",
    # TimeLimit args
    max_episode_steps=200,
    # SequentialTask args
    task_plans=plan_data.plans,
    scene_builder_cls=plan_data.dataset,
    # SubtaskTrain args
    spawn_data_fp=spawn_data_fp,
    # optional: additional env_kwargs
    sensor_configs=dict(width=320, height=240),
    human_render_camera_configs=dict(shader_pack="rt"),
    viewer_camera_configs=dict(fov=1),
)

# print(plan_data)


obs, _ = env.reset(seed=0) # reset with a seed for determinism
done = False
for i in range(20000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    if i ==1 :#"subtask" in info:  # info 딕셔너리에 subtask 정보가 있는지 확인
        print(f"Current Subtask (from env): {info}")
    elif i == 0: # 처음 한 번만 출력
        print("Subtask information not found in 'info' dictionary.")

    env.render()  # a display is required to render
env.close()


# # add env wrappers here

# venv = ManiSkillVectorEnv(
#     env,
#     max_episode_steps=1000,  # set manually based on task
#     ignore_terminations=True,  # set to False for partial resets
# )


# # add vector env wrappers here

# obs, info = venv.reset()

