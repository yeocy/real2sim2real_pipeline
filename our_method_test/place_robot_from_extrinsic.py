"""
place_robot_from_extrinsic.py — Step 3 씬에 원본(LIBERO) 로봇을 카메라 extrinsic 기준으로 배치하고 찍는다.

Step 3 는 물체를 카메라 기준으로 배치하므로(씬 좌표계는 Step 1 지지면 기준으로 새로 잡힌다),
원본 시뮬레이터의 '카메라 <-> 로봇' 상대 pose 를 씬 카메라에 그대로 붙이면 로봇 자리가 나온다.

  T_cam_robot   = inv(T_world_cam^LIBERO) @ T_world_link0^LIBERO
  T_scene_robot = T_scene_cam @ T_cam_robot          (T_scene_cam = scene_info["cam_pose"])

두 카메라 모두 opengl 규약(-z 전방, +y 위)이라 그대로 곱한다. 로봇은 OG FrankaPanda 로,
root(panda_link0) 를 LIBERO robot0_link0 pose 에 두고 관절도 LIBERO 초기값으로 맞춘다.
렌더는 원본 카메라와 같은 해상도·화각(fovy)으로 찍어 원본 사진과 나란히 저장한다.
씬 폴더(scene_info 옆)에 task_info.json 을 남긴다: language instruction, 씬 좌표계의 로봇 pose
(position, quaternion xyzw/wxyz), 관절 초기값, 카메라 pose.

입력 robot pose json (LIBERO 에서 추출):
  {"camera": {position, orientation(xyzw)}, "robot0_link0": {...}, "joints": {robot0_joint1.., gripper0_finger_joint1..}}

실행 (acdc env, PYTHONPATH 에 airlab_twin):
  python place_robot_from_extrinsic.py --scene_info <step_3_output/scene_0/scene_0_info.json> \
      --robot_pose <libero_robot_pose.json> --camera_info <inputs/.../camera_info.json> \
      --pool asset_pools_local/libero10_04 --reference <libero_with_robot.png> --out <dir>
"""
import os
import sys
import json
import math
import argparse

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")

import cv2                                      # og.launch() 전에 import (Isaac 번들 PIL 충돌 방지)
import numpy as np
import torch as th
from PIL import Image
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from our_method.utils.asset_pool import use_pool_usd  # noqa: E402


def mat(pose):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(pose["orientation"]).as_matrix()
    T[:3, 3] = pose["position"]
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_info", required=True)
    ap.add_argument("--robot_pose", required=True)
    ap.add_argument("--camera_info", required=True, help="원본 카메라 해상도/화각")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--reference", default=None, help="원본 시뮬레이터에서 로봇을 포함해 찍은 같은 카메라 사진")
    ap.add_argument("--out", required=True)
    ap.add_argument("--language", default=None,
                    help="language instruction. 안 주면 camera_info.json 의 task_description 을 쓴다")
    args = ap.parse_args()

    scene_info = json.load(open(args.scene_info))
    rp = json.load(open(args.robot_pose))
    ci = json.load(open(args.camera_info))
    W, H = ci["intrinsics"]["image_width"], ci["intrinsics"]["image_height"]
    language = args.language or ci.get("task_description")
    fovy = ci["intrinsics"].get("fovy_deg", 45.0)

    cam_pos, cam_quat = scene_info["cam_pose"]
    T_scene_cam = mat({"position": cam_pos, "orientation": cam_quat})
    T_cam_robot = np.linalg.inv(mat(rp["camera"])) @ mat(rp["robot0_link0"])
    T_scene_robot = T_scene_cam @ T_cam_robot
    robot_pos = T_scene_robot[:3, 3]
    robot_quat = Rotation.from_matrix(T_scene_robot[:3, :3]).as_quat()
    tilt = math.degrees(math.acos(np.clip(T_scene_robot[2, 2], -1, 1)))
    print(f"[robot] scene pose pos={robot_pos.round(4)} quat={robot_quat.round(4)}  "
          f"(z축 기울기 {tilt:.2f}도, 0 이면 씬 바닥과 평행)")

    import omnigibson as og
    from omnigibson.robots import FrankaPanda
    from our_method.utils.physics_settle import load_scene

    og.launch()
    use_pool_usd(args.pool)
    scene = load_scene(scene_info, "static", [])
    with og.sim.stopped():
        robot = FrankaPanda(name="robot0", fixed_base=True, obs_modalities=[])
        scene.add_object(robot)
    og.sim.play()
    robot.set_position_orientation(th.tensor(robot_pos, dtype=th.float), th.tensor(robot_quat, dtype=th.float))
    j = rp["joints"]
    arm = [j[f"robot0_joint{i}"] for i in range(1, 8)]
    fingers = [abs(j.get("gripper0_finger_joint1", 0.04)), abs(j.get("gripper0_finger_joint2", 0.04))]
    idx = [robot.joints[n].joint_idx if hasattr(robot.joints[n], "joint_idx") else list(robot.joints).index(n)
           for n in robot.arm_joint_names[robot.default_arm] + robot.finger_joint_names[robot.default_arm]]
    robot.set_joint_positions(th.tensor(arm + fingers, dtype=th.float), indices=th.tensor(idx))
    og.sim.step()

    cam = og.sim.viewer_camera
    cam.image_width, cam.image_height = W, H
    # 정사각/직사각 모두 세로 화각을 원본과 맞춘다: fovy = 2 atan(ap_v / 2f), ap_v = ap_h * H / W
    f = cam.focal_length
    cam.horizontal_aperture = 2 * f * math.tan(math.radians(fovy) / 2) * W / H
    cam.set_position_orientation(th.tensor(cam_pos, dtype=th.float), th.tensor(cam_quat, dtype=th.float))
    for _ in range(30):
        og.sim.render()
    rgb = cam.get_obs()[0]["rgb"][:, :, :3].cpu().numpy()
    rgb = rgb if rgb.dtype == np.uint8 else (rgb * 255).astype(np.uint8)

    os.makedirs(args.out, exist_ok=True)
    Image.fromarray(rgb).save(os.path.join(args.out, "twin_with_robot.png"))
    json.dump({"robot_pos": robot_pos.tolist(), "robot_quat_xyzw": robot_quat.tolist(),
               "T_cam_robot": T_cam_robot.tolist(), "tilt_deg": tilt},
              open(os.path.join(args.out, "robot_scene_pose.json"), "w"), indent=2)
    # 씬 폴더에 태스크/로봇 정보 기록 (씬 좌표계 = scene_info 의 좌표계)
    task_info = {
        "language_instruction": language,
        "scene_info": os.path.basename(args.scene_info),   # 같은 폴더 기준 상대경로 (폴더를 옮겨도 된다)
        "frame": "twin scene world frame (scene_info cam_pose 와 같은 좌표계, z-up, m)",
        "robot": {
            "type": "FrankaPanda (OmniGibson), root link = panda_link0",
            "position": [float(x) for x in robot_pos],
            "quaternion_xyzw": [float(x) for x in robot_quat],
            "quaternion_wxyz": [float(robot_quat[3]), float(robot_quat[0]), float(robot_quat[1]), float(robot_quat[2])],
            "arm_joint_positions": [float(x) for x in arm],
            "finger_joint_positions": [float(x) for x in fingers],
            "source": "LIBERO robot0_link0 pose, mapped by camera extrinsic (T_scene_cam @ inv(T_world_cam) @ T_world_link0)",
        },
        "camera": {"position": [float(x) for x in cam_pos], "quaternion_xyzw": [float(x) for x in cam_quat],
                   "convention": "opengl (-z forward, +y up)", "image_width": W, "image_height": H, "fovy_deg": fovy},
    }
    task_path = os.path.join(os.path.dirname(os.path.abspath(args.scene_info)), "task_info.json")
    json.dump(task_info, open(task_path, "w"), indent=2, ensure_ascii=False)
    print(f"[robot] task info -> {task_path}")
    if args.reference:
        ref = cv2.imread(args.reference)
        twin = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ref = cv2.resize(ref, (twin.shape[1], twin.shape[0]))
        blend = cv2.addWeighted(ref, 0.5, twin, 0.5, 0)
        for im, t in [(ref, "LIBERO (original)"), (twin, "twin + robot (extrinsic)"), (blend, "overlay 50/50")]:
            cv2.putText(im, t, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(args.out, "compare.png"), np.hstack([ref, twin, blend]))
    print(f"[robot] saved -> {args.out}")
    og.shutdown()


if __name__ == "__main__":
    main()
