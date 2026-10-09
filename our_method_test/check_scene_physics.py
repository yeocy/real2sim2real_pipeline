"""
최종 씬(scene_info.json)이 물리적으로 성립하는지 OmniGibson(headless)에서 잰다.

1. 정적 측정 (모든 물체 visual_only + fixed_base 로 로드해 아무것도 움직이지 않게 한 상태)
   - 물체마다 충돌 메시 AABB, 시각 메시 AABB
   - 메타데이터 bbox 중심(set_bbox_center_position_orientation 이 기준으로 쓰는 점)과 실제 충돌 AABB 중심의 차이
   - 받침 물체(scene graph 의 objBeneath) 기준 관통 깊이 = 받침 윗면 - 내 아랫면 (양수면 파고듦)
2. 동적 측정 (충돌을 켜고 fixed_categories 만 고정, 나머지는 동적 물체)
   - 첫 스텝 직후 물체 쌍 접촉(Touching)
   - 중력으로 N 스텝 돌렸을 때 저장된 포즈 대비 이동량 (xy, z, yaw, tilt)

사용:
    python check_scene_physics.py <scene_info.json> [--steps 300] [--out report.json]
"""
import os
import sys
import json
import argparse

os.environ["OMNIGIBSON_HEADLESS"] = "1"

import numpy as np
import torch as th
import omnigibson as og
from our_method.utils.asset_pool import use_pool_usd, DEFAULT_POOL_ROOT  # launch 전에 import (PIL 충돌)
from omnigibson.object_states import Touching

from our_method.utils.physics_settle import (np_, world_pose, yaw_tilt, wrap, supports_of, load_scene,
                                             to_builtin, PHYSICS_SETTLE_DEFAULTS)
import our_method.utils.transform_utils as T

DEFAULT_FIXED_CATEGORIES = tuple(PHYSICS_SETTLE_DEFAULTS["fixed_categories"])


def static_metrics(scene, scene_info):
    res = {}
    for name in scene_info["objects"]:
        obj = scene.object_registry("name", name)
        pos, quat = (np_(v) for v in obj.get_position_orientation())
        c_lo, c_hi = (np_(v) for v in obj.aabb)
        vis = [np_(l.visual_boundary_points_world) for l in obj.links.values()
               if l.visual_boundary_points_world is not None]
        vis = np.concatenate(vis, axis=0) if vis else np.stack([c_lo, c_hi])
        v_lo, v_hi = vis.min(axis=0), vis.max(axis=0)
        meta_center = pos + T.quat2mat(quat) @ np_(obj.scaled_bbox_center_in_base_frame)
        res[name] = {
            "pos": pos, "quat": quat,
            "coll_lo": c_lo, "coll_hi": c_hi, "vis_lo": v_lo, "vis_hi": v_hi,
            # set_bbox_center_position_orientation 이 "bbox 중심"이라고 믿는 점과 실제 충돌 AABB 중심의 차이
            "meta_center_err": (c_lo + c_hi) / 2 - meta_center,
        }
    return res


def penetrations(st, sup, scene_info):
    """받침 물체 기준 관통 깊이(양수 = 파고듦). 그릇 안 내용물은 AABB 로 잴 수 없어 None."""
    out = {}
    for name, m in st.items():
        s = sup.get(name)
        info = scene_info["objects"][name]
        if info.get("placement") == "inside":
            out[name] = (s, None, None)
            continue
        if s is None or s == "floor" or s not in st:
            top_c = top_v = 0.0
            s = "floor"
        else:
            top_c, top_v = st[s]["coll_hi"][2], st[s]["vis_hi"][2]
        out[name] = (s, float(top_c - m["coll_lo"][2]), float(top_v - m["vis_lo"][2]))
    return out


def touching_pairs(scene, names):
    pairs = []
    objs = [scene.object_registry("name", n) for n in names]
    for i, a in enumerate(objs):
        for b in objs[i + 1:]:
            if a.states[Touching].get_value(b):
                pairs.append((a.name, b.name))
    return pairs


def motion(scene, scene_info, steps):
    cam_pose = scene_info["cam_pose"]
    ref = {n: world_pose(i, cam_pose) for n, i in scene_info["objects"].items()}
    og.sim.step()
    pairs = touching_pairs(scene, list(scene_info["objects"].keys()))
    for _ in range(steps - 1):
        og.sim.step()
    out = {}
    for name, (p0, q0) in ref.items():
        obj = scene.object_registry("name", name)
        p1, q1 = (np_(v) for v in obj.get_position_orientation())
        y0, t0 = yaw_tilt(q0)
        y1, t1 = yaw_tilt(q1)
        out[name] = {
            "dxy": float(np.linalg.norm(p1[:2] - np.asarray(p0[:2]))),
            "dz": float(p1[2] - p0[2]),
            "dyaw_deg": float(np.rad2deg(abs(wrap(y1 - y0)))),
            "dtilt_deg": float(np.rad2deg(abs(t1 - t0))),
            "final_pos": p1,
        }
    return out, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_info")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--fixed_categories", nargs="*", default=list(DEFAULT_FIXED_CATEGORIES))
    ap.add_argument("--out", default=None, help="json 리포트 경로 (기본: scene_info 옆 physics_check.json)")
    ap.add_argument("--xy_tol", type=float, default=0.01)
    ap.add_argument("--z_tol", type=float, default=0.005)
    ap.add_argument("--yaw_tol", type=float, default=3.0)
    ap.add_argument("--pen_tol", type=float, default=0.003)
    ap.add_argument("--usd_pool", default=None,
                    help="USD 를 먼저 불러올 풀 (기본: 통합 풀 kist_twin). 'none' 이면 OG 데이터셋만 쓴다")
    args = ap.parse_args()

    with open(args.scene_info) as f:
        scene_info = json.load(f)
    sup = supports_of(scene_info)

    og.launch()
    if (args.usd_pool or "").lower() != "none":
        use_pool_usd(args.usd_pool or DEFAULT_POOL_ROOT)

    scene = load_scene(scene_info, "static", args.fixed_categories)
    og.sim.step()
    st = static_metrics(scene, scene_info)
    pen = penetrations(st, sup, scene_info)

    scene = load_scene(scene_info, "dynamic", args.fixed_categories)
    mv, pairs = motion(scene, scene_info, args.steps)

    print("\n" + "=" * 120)
    print(f"scene: {args.scene_info}")
    print(f"{'object':24s} {'support':20s} {'pen_coll':>9s} {'pen_vis':>8s} {'meta_dz':>8s} "
          f"{'dxy':>7s} {'dz':>8s} {'dyaw':>6s} {'dtilt':>6s}  flags")
    n_bad = 0
    for name in scene_info["objects"]:
        s, pc, pv = pen[name]
        m = mv[name]
        flags = []
        if pc is not None and pc > args.pen_tol:
            flags.append("PEN")
        if m["dxy"] > args.xy_tol or abs(m["dz"]) > args.z_tol or m["dyaw_deg"] > args.yaw_tol:
            flags.append("MOVED")
        n_bad += bool(flags)
        f_ = lambda v, w=8, p=4: f"{v:{w}.{p}f}" if v is not None else f"{'-':>{w}s}"
        print(f"{name:24s} {str(s):20s} {f_(pc, 9)} {f_(pv)} {f_(st[name]['meta_center_err'][2])} "
              f"{f_(m['dxy'], 7)} {f_(m['dz'])} {m['dyaw_deg']:6.2f} {m['dtilt_deg']:6.2f}  {' '.join(flags)}")
    print(f"touching pairs after 1 step ({len(pairs)}): {pairs}")
    print(f"{n_bad} / {len(scene_info['objects'])} objects flagged "
          f"(pen>{args.pen_tol * 1000:.0f}mm, xy>{args.xy_tol * 100:.0f}cm, z>{args.z_tol * 1000:.0f}mm, "
          f"yaw>{args.yaw_tol:.0f}deg over {args.steps} steps)")
    print("=" * 120 + "\n")

    report = {
        "scene_info": os.path.abspath(args.scene_info), "steps": args.steps,
        "fixed_categories": args.fixed_categories, "touching_pairs": pairs, "n_flagged": n_bad,
        "objects": {n: {"support": pen[n][0], "pen_coll": pen[n][1], "pen_vis": pen[n][2],
                        **{k: st[n][k] for k in ("coll_lo", "coll_hi", "vis_lo", "vis_hi", "meta_center_err")},
                        **mv[n]} for n in scene_info["objects"]},
    }
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.scene_info)), "physics_check.json")
    with open(out, "w") as f:
        json.dump(to_builtin(report), f, indent=2)
    print(f"report -> {out}")
    sys.stdout.flush()
    og.shutdown()


if __name__ == "__main__":
    main()
