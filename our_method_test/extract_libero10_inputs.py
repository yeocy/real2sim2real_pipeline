"""
LIBERO-10 의 10개 task 각각에 대해 airlab_twin 파이프라인 입력을 뽑는다.

출력 (inputs_kist_twin_full 과 같은 형식):
  <out>/<NN>_<bddl 이름>/
    camera_rgb.png       agentview RGB (H x W)
    camera_depth.npy     metric depth (m), (H, W, 1)
    camera_depth.png     depth 정규화 시각화
    camera_info.json     카메라 pose (opengl, xyzw) + intrinsics + task_description
    scene_info.json      body 별 world pose
camera_rgb_resize.png 는 파이프라인이 만든다.

물체가 안정된 뒤 로봇(robot0_base 와 그 하위 mount/gripper)을 바닥 아래 멀리 옮기고
(물리 step 없이 forward 만), 씬의 모든 object/fixture geom 이 화면에 들어오도록
agentview 카메라 pose 를 자동으로 다시 잡아서 찍는다 (--elev, --azim, --margin).
init state 는 benchmark 의 기본 init_states[idx].

실행 (libero conda env):
  LIBERO_CONFIG_PATH=<config 디렉토리> PYTHONPATH=<원본 LIBERO 경로> \
  MUJOCO_GL=egl python extract_libero10_inputs.py --out inputs_libero_10
"""

import argparse
import json
import os
import re

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def depth_to_meters(sim, depth):
    """robosuite 정규화 z-buffer -> metric depth."""
    extent = sim.model.stat.extent
    near = sim.model.vis.map.znear * extent
    far = sim.model.vis.map.zfar * extent
    return near / (1 - depth * (1 - near / far))


def intrinsics_from_fovy(fovy_deg, width, height):
    """MuJoCo fovy(수직 화각) -> 3x3 K. 정사각 픽셀이라 fx == fy."""
    fy = 0.5 * height / np.tan(0.5 * np.deg2rad(fovy_deg))
    return np.array([[fy, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]])


def hide_robot(sim, offset=(0.0, 0.0, -50.0)):
    """로봇 root body 들을 멀리 옮긴다. step 하지 않으므로 물체 상태는 그대로."""
    for bid in range(sim.model.nbody):
        name = sim.model.body_id2name(bid) or ""
        if sim.model.body_parentid[bid] == 0 and name.startswith(("robot", "mount", "gripper")):
            sim.model.body_pos[bid] += np.array(offset)
    sim.forward()


def scene_spheres(sim, root_bodies):
    """root_bodies 아래 모든 geom 의 (world 중심, bounding radius)."""
    roots = {sim.model.body_name2id(b) for b in root_bodies}
    centers, radii = [], []
    for g in range(sim.model.ngeom):
        b = sim.model.geom_bodyid[g]
        while b not in roots and b != 0:
            b = sim.model.body_parentid[b]
        if b in roots:
            centers.append(sim.data.geom_xpos[g].copy())
            radii.append(float(sim.model.geom_rbound[g]))
    return np.array(centers), np.array(radii)


def look_at_rotation(cam_pos, target):
    """opengl 카메라(-z forward, +y up) world 회전행렬."""
    f = target - cam_pos
    f /= np.linalg.norm(f)
    x = np.cross(f, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return np.stack([x, y, -f], axis=1)


def frame_camera(sim, cam_id, centers, radii, elev_deg, azim_deg, margin):
    """모든 bounding sphere 가 화면 안(반화각 * margin)에 들어오는 최소 거리로 카메라 배치."""
    lo = (centers - radii[:, None]).min(0)
    hi = (centers + radii[:, None]).max(0)
    target = (lo + hi) / 2.0
    e, a = np.deg2rad(elev_deg), np.deg2rad(azim_deg)
    back = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])  # target -> camera
    tan_half = np.tan(np.deg2rad(sim.model.cam_fovy[cam_id]) / 2.0) * margin

    def fits(dist):
        pos = target + back * dist
        Rm = look_at_rotation(pos, target)
        pc = (centers - pos) @ Rm  # camera frame
        depth = -pc[:, 2] - radii
        if (depth <= 0.05).any():
            return False
        return ((np.abs(pc[:, :2]) + radii[:, None]) / depth[:, None] <= tan_half).all()

    near, far = 0.1, 20.0
    for _ in range(50):
        mid = (near + far) / 2.0
        near, far = (near, mid) if fits(mid) else (mid, far)
    pos = target + back * far
    quat = Rotation.from_matrix(look_at_rotation(pos, target)).as_quat()  # xyzw
    sim.model.cam_pos[cam_id] = pos
    sim.model.cam_quat[cam_id] = [quat[3], quat[0], quat[1], quat[2]]  # mujoco wxyz
    sim.forward()


def extract_task(task_suite, task_id, args):
    task = task_suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    out_dir = os.path.join(args.out, f"{task_id:02d}_{os.path.splitext(task.bddl_file)[0]}")
    os.makedirs(out_dir, exist_ok=True)

    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_names=[args.cam],
        camera_heights=args.res,
        camera_widths=args.res,
        camera_depths=True,
    )
    env.seed(0)
    env.reset()
    init_states = task_suite.get_task_init_states(task_id)
    obs = env.set_init_state(init_states[args.init_idx])
    # LIBERO eval 과 같이 물체가 안정될 때까지 대기 (팔 고정, 그리퍼 open)
    for _ in range(args.settle_steps):
        obs, _, _, _ = env.step([0.0] * 6 + [-1.0])

    sim = env.sim
    cam_id = sim.model.camera_name2id(args.cam)
    if not args.keep_robot:
        hide_robot(sim)
    if not args.keep_cam:
        inner = env.env
        roots = [o.root_body for name, o in {**inner.objects_dict, **inner.fixtures_dict}.items()
                 if not name.endswith("table")]
        centers, radii = scene_spheres(sim, roots)
        frame_camera(sim, cam_id, centers, radii, args.elev, args.azim, args.margin)

    # robosuite(opengl) 렌더는 상하 반전 -> flipud 로 일반 이미지 좌표계로
    rgb, zbuf = sim.render(width=args.res, height=args.res, camera_name=args.cam, depth=True)
    rgb = np.ascontiguousarray(np.flipud(rgb))
    depth = np.flipud(depth_to_meters(sim, zbuf))[..., None].astype(np.float32)

    cv2.imwrite(os.path.join(out_dir, "camera_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save(os.path.join(out_dir, "camera_depth.npy"), depth)
    lo, hi = float(depth.min()), float(depth.max())
    cv2.imwrite(
        os.path.join(out_dir, "camera_depth.png"),
        ((depth - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8),
    )

    pos = sim.data.cam_xpos[cam_id]
    quat = Rotation.from_matrix(sim.data.cam_xmat[cam_id].reshape(3, 3)).as_quat()  # xyzw
    fovy = float(sim.model.cam_fovy[cam_id])
    K = intrinsics_from_fovy(fovy, args.res, args.res)

    camera_info = {
        "source": "libero",
        "task_suite": "libero_10",
        "task_id": task_id,
        "task_name": task.name,
        "bddl_file": bddl,
        "init_state_idx": args.init_idx,
        "robot_removed": not args.keep_robot,
        "task_description": task.language,
        "camera": {
            "position": [float(v) for v in pos],
            "orientation": [float(v) for v in quat],
            "orientation_format": "xyzw",
            "convention": "opengl (-z forward, +y up)",
            "name": args.cam,
        },
        "intrinsics": {
            "image_width": args.res,
            "image_height": args.res,
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "fovy_deg": fovy,
            "matrix": K.tolist(),
        },
        "depth": {
            "unit": "meter",
            "min": lo,
            "max": hi,
            "mean": float(depth.mean()),
            "shape": list(depth.shape),
        },
    }
    with open(os.path.join(out_dir, "camera_info.json"), "w") as f:
        json.dump(camera_info, f, indent=2)

    scene_info = {"task_description": task.language, "objects": {}}
    for bid in range(sim.model.nbody):
        name = sim.model.body_id2name(bid)
        if not name or name == "world" or name.startswith(("robot", "mount", "gripper")):
            continue
        q = sim.data.body_xquat[bid]  # wxyz
        scene_info["objects"][name] = {
            "position": [float(v) for v in sim.data.body_xpos[bid]],
            "orientation": [float(q[1]), float(q[2]), float(q[3]), float(q[0])],
            "orientation_format": "xyzw",
        }
    with open(os.path.join(out_dir, "scene_info.json"), "w") as f:
        json.dump(scene_info, f, indent=2)

    env.close()
    print(f"[{task_id}] {task.language}\n    -> {out_dir}  depth {lo:.3f}~{hi:.3f} m")
    return {"task_id": task_id, "task_description": task.language, "dir": os.path.basename(out_dir)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="inputs_libero_10")
    ap.add_argument("--cam", default="agentview")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--init_idx", type=int, default=0)
    ap.add_argument("--settle_steps", type=int, default=10)
    ap.add_argument("--keep_robot", action="store_true", help="로봇을 치우지 않음")
    ap.add_argument("--keep_cam", action="store_true", help="기본 agentview pose 그대로")
    ap.add_argument("--elev", type=float, default=45.0, help="카메라 앙각(deg)")
    ap.add_argument("--azim", type=float, default=0.0, help="카메라 방위각(deg), 0 = +x 쪽(agentview 방향)")
    ap.add_argument("--margin", type=float, default=0.9, help="반화각 대비 사용 비율")
    ap.add_argument("--tasks", type=int, nargs="*", default=None, help="기본: 전체 10개")
    args = ap.parse_args()

    task_suite = benchmark.get_benchmark_dict()["libero_10"]()
    ids = args.tasks if args.tasks else range(task_suite.n_tasks)
    index = [extract_task(task_suite, i, args) for i in ids]
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "tasks.json"), "w") as f:
        json.dump(index, f, indent=2)


if __name__ == "__main__":
    main()
