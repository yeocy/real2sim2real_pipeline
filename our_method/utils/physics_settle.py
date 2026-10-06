"""
최종 씬을 물리적으로 성립하는 상태로 만든다 (관통 제거 + 중력 안착). Step 3 와 Step 7 이 저장 직전에 부른다.

settle_scene(scene, scene_info, cfg) 순서:
1. 관통 해소: 받침 물체(scene graph objBeneath / parent_object) 위에 놓인 물체를 아래에서 위 순서
   (테이블 -> 인덕션 -> 냄비)로 돌며, 충돌 AABB 아랫면이 받침 AABB 윗면 + lift_margin_m 에 오도록 올리거나 내린다.
   위에 얹힌 물체(자손)도 같은 만큼 함께 움직인다. 그릇 안 내용물(placement == "inside")은 AABB 로 높이를 정할 수
   없으므로 부모를 따라 움직이기만 한다. 이 단계는 시뮬레이션을 돌리지 않고 AABB 만 쓴다.
   (Touching 으로 접촉을 찾으며 내리는 방식은 접촉이 늦게 잡혀 수 cm 씩 파고들었다.)
2. 중력 안착: 씬을 새로 로드해 fixed_categories(테이블 등)와 벽 부착 물체만 fixed_base 로 고정하고,
   나머지는(내용물 포함) 동적 물체로 두고 steps 만큼 시뮬레이션한다.
3. 안정성 검사: check_steps 만큼 더 돌려 이동량이 stable_* 이하인지 본다. 크게 움직인 물체는 로그로 남긴다.
4. 원래 위치 유지: 안착 전(1 단계 후) 대비 xy, yaw 변화를 기록한다. xy_tol_m / yaw_tol_deg 를 넘으면 경고하고,
   revert_xy_yaw 가 켜져 있으면 그 물체는 안착 전 xy·자세로 되돌리고 높이만 안착 결과를 쓴다 (내용물 제외).
5. 결과 기록: 최종 포즈로 scene_info["objects"][*]["tf_from_cam"] 을 갱신한다.

반환: (안착이 끝난 새 scene, report dict). 새 scene 은 렌더링(사진, 영상)에만 쓰고 더 스텝하지 않는다고 가정한다.
"""
import os
import json
from collections import deque

import numpy as np
import torch as th
from loguru import logger as log

import omnigibson as og
from omnigibson.objects import DatasetObject

from our_method.utils.scene_utils import create_scene
import our_method.utils.transform_utils as T

PHYSICS_SETTLE_DEFAULTS = {
    "enabled": False,
    "steps": 300,                 # 중력 안착 스텝 수
    "check_steps": 150,           # 안정성 검사 스텝 수
    "fixed_categories": ["white_table"],
    "lift_margin_m": 0.001,       # 관통 해소 후 받침과의 간격
    "xy_tol_m": 0.02,             # 안착 전후 xy 이동 허용치
    "yaw_tol_deg": 5.0,           # 안착 전후 yaw 변화 허용치
    "revert_xy_yaw": True,        # 허용치를 넘으면 xy·자세는 안착 전으로, 높이만 안착 결과로
    "stable_xy_m": 0.01,          # 안정성 검사 중 허용 이동량
    "stable_z_m": 0.005,
    "stable_yaw_deg": 3.0,
    "tilt_warn_deg": 30.0,        # 안착 중 이보다 크게 기울면(넘어짐/세워짐) 경고
}


def resolve_cfg(cfg):
    """config 값(None / bool / dict)을 기본값과 합친다. 꺼져 있으면 None."""
    if cfg is None or cfg is False:
        return None
    if cfg is True:
        cfg = {"enabled": True}
    cfg = {**PHYSICS_SETTLE_DEFAULTS, **dict(cfg)}
    return cfg if cfg["enabled"] else None


def np_(x):
    return x.detach().cpu().numpy() if isinstance(x, th.Tensor) else np.asarray(x)


def world_pose(obj_info, cam_pose):
    """tf_from_cam -> 월드 (pos, quat)."""
    return T.mat2pose(T.pose_in_A_to_pose_in_B(pose_A=np.array(obj_info["tf_from_cam"]),
                                               pose_A_in_B=T.pose2mat(cam_pose)))


def tf_from_cam(pos, quat, cam_pose):
    return T.pose2mat(T.relative_pose_transform(np.asarray(pos), np.asarray(quat),
                                                np.asarray(cam_pose[0]), np.asarray(cam_pose[1])))


def yaw_tilt(quat):
    """(yaw, tilt) [rad]. tilt 은 물체 z 축이 월드 z 축에서 기운 각도."""
    m = T.quat2mat(np.asarray(quat))
    return float(np.arctan2(m[1, 0], m[0, 0])), float(np.arccos(np.clip(m[2, 2], -1.0, 1.0)))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def supports_of(scene_info):
    """물체 -> 받침 물체 이름. scene graph(Step 3)의 objBeneath 와 parent_object(Step 7 물체)를 합친다."""
    sup = {}
    graph_path = scene_info.get("scene_graph")
    if graph_path and os.path.exists(graph_path):
        with open(graph_path) as f:
            graph = json.load(f)
        for name, g in graph.items():
            if name in scene_info["objects"]:
                sup[name] = g.get("objBeneath")
    for name, info in scene_info["objects"].items():
        if info.get("parent_object"):
            sup[name] = info["parent_object"]
    return sup


def is_fixed(info, fixed_categories):
    mount = info.get("mount") or {}
    return info["category"] in fixed_categories or bool(mount.get("wall"))


def is_inside(info):
    return info.get("placement") == "inside"


def load_scene(scene_info, mode, fixed_categories):
    """
    scene_info 를 새 씬으로 로드한다.
    mode="static": 전부 visual_only + fixed_base (AABB 측정용, 아무것도 움직이지 않는다)
    mode="dynamic": 충돌을 켜고 fixed_categories / 벽 부착 물체만 fixed_base
    """
    scene = create_scene(floor=True)
    cam_pose = scene_info["cam_pose"]
    og.sim.viewer_camera.set_position_orientation(th.tensor(cam_pose[0], dtype=th.float),
                                                  th.tensor(cam_pose[1], dtype=th.float))
    with og.sim.stopped():
        for name, info in scene_info["objects"].items():
            obj = DatasetObject(
                name=name, category=info["category"], model=info["model"], scale=info["scale"],
                visual_only=(mode == "static"),
                fixed_base=(mode == "static" or is_fixed(info, fixed_categories)),
            )
            scene.add_object(obj)
            pos, quat = world_pose(info, cam_pose)
            obj.set_position_orientation(th.tensor(pos, dtype=th.float), th.tensor(quat, dtype=th.float))
    return scene


def _support_order(names, sup):
    """받침 물체가 먼저 오도록 정렬 (깊이 순). 순환이 있으면 남은 것을 뒤에 붙인다."""
    children = {n: [] for n in names}
    roots = []
    for n in names:
        s = sup.get(n)
        if s in children and s != n:
            children[s].append(n)
        else:
            roots.append(n)
    order, q = [], deque(roots)
    while q:
        n = q.popleft()
        order.append(n)
        q.extend(children[n])
    order += [n for n in names if n not in order]
    return order, children


def _descendants(name, children):
    out, q = [], deque(children.get(name, []))
    while q:
        n = q.popleft()
        if n not in out and n != name:
            out.append(n)
            q.extend(children.get(n, []))
    return out


def resolve_penetration(scene, scene_info, sup, cfg):
    """
    1 단계. AABB 로 받침 물체 윗면에 맞춘다. scene 은 스텝하지 않는다 (visual_only 이든 fixed_base 이든 상관없음).
    반환: {name: {"support", "gap_before", "dz"}}
    """
    names = list(scene_info["objects"].keys())
    order, children = _support_order(names, sup)
    margin = cfg["lift_margin_m"]
    res = {}
    for name in order:
        info = scene_info["objects"][name]
        if is_inside(info):
            continue
        obj = scene.object_registry("name", name)
        s = sup.get(name)
        lo = np_(obj.aabb[0])[2]
        if s in scene_info["objects"] and s != name:
            top = np_(scene.object_registry("name", s).aabb[1])[2]
        else:
            s, top = "floor", 0.0
        target_gap = 0.0 if (s == "floor" or is_fixed(info, cfg["fixed_categories"])) else margin
        gap = float(lo - top)
        dz = target_gap - gap
        res[name] = {"support": s, "gap_before": gap, "dz": dz}
        if abs(dz) < 1e-5:
            continue
        for n in [name] + _descendants(name, children):
            o = scene.object_registry("name", n)
            p, q = o.get_position_orientation()
            o.set_position_orientation(position=p + th.tensor([0.0, 0.0, dz], dtype=p.dtype, device=p.device),
                                       orientation=q)
        if abs(gap) > 0.002:
            log.info(f"[physics_settle] {name} on {s}: 간격 {gap * 1000:+.1f}mm -> {target_gap * 1000:+.1f}mm "
                     f"(dz {dz * 1000:+.1f}mm, 위에 얹힌 {len(_descendants(name, children))}개 함께 이동)")
    return res


def _poses(scene, names):
    return {n: tuple(np_(v) for v in scene.object_registry("name", n).get_position_orientation()) for n in names}


def _delta(p0, p1):
    (a, qa), (b, qb) = p0, p1
    ya, ta = yaw_tilt(qa)
    yb, tb = yaw_tilt(qb)
    return {
        "dxy": float(np.linalg.norm(b[:2] - a[:2])),
        "dz": float(b[2] - a[2]),
        "dyaw_deg": float(np.rad2deg(abs(wrap(yb - ya)))),
        "dtilt_deg": float(np.rad2deg(abs(tb - ta))),
    }


def _inside_check(scene, scene_info, names):
    """그릇 안 내용물이 여전히 부모 안에 있는지 (중심이 부모 반폭 안, 바닥이 림보다 아래)."""
    out = {}
    for n in names:
        info = scene_info["objects"][n]
        if not is_inside(info) or info.get("parent_object") not in scene_info["objects"]:
            continue
        p = scene.object_registry("name", info["parent_object"])
        lo, hi = (np_(v) for v in p.aabb)
        half = min(hi[0] - lo[0], hi[1] - lo[1]) / 2
        o = scene.object_registry("name", n)
        pos = np_(o.get_position_orientation()[0])
        clo = np_(o.aabb[0])
        r = float(np.linalg.norm(pos[:2] - (lo[:2] + hi[:2]) / 2))
        out[n] = {"r_xy": r, "parent_half": float(half), "inside": bool(r <= half and clo[2] < hi[2])}
    return out


def settle_scene(scene, scene_info, cfg):
    """
    @scene 의 물체로 관통을 풀고, 씬을 새로 로드해 중력으로 안착시킨 뒤 scene_info 의 tf_from_cam 을 갱신한다.
    @cfg 는 resolve_cfg 를 거친 dict. 반환: (새 scene, report)
    """
    cam_pose = scene_info["cam_pose"]
    names = list(scene_info["objects"].keys())
    sup = supports_of(scene_info)
    fixed_cats = cfg["fixed_categories"]
    report = {"cfg": cfg, "objects": {n: {} for n in names}, "warnings": []}

    # 1. 관통 해소 (AABB)
    lift = resolve_penetration(scene, scene_info, sup, cfg)
    for n in names:
        scene_info["objects"][n]["tf_from_cam"] = tf_from_cam(
            *(np_(v) for v in scene.object_registry("name", n).get_position_orientation()), cam_pose)
        report["objects"][n]["lift"] = lift.get(n)

    # 2. 중력 안착
    scene = load_scene(scene_info, "dynamic", fixed_cats)
    pre = {n: world_pose(scene_info["objects"][n], cam_pose) for n in names}
    for n in names:
        scene.object_registry("name", n).keep_still()
    for _ in range(cfg["steps"]):
        og.sim.step()
    mid = _poses(scene, names)

    # 3. 안정성 검사
    for _ in range(cfg["check_steps"]):
        og.sim.step()
    post = _poses(scene, names)

    for n in names:
        info = scene_info["objects"][n]
        r = report["objects"][n]
        r["settle"] = _delta(pre[n], mid[n])
        r["check"] = _delta(mid[n], post[n])
        r["stable"] = (r["check"]["dxy"] <= cfg["stable_xy_m"] and abs(r["check"]["dz"]) <= cfg["stable_z_m"]
                       and r["check"]["dyaw_deg"] <= cfg["stable_yaw_deg"])
        total = _delta(pre[n], post[n])
        r["total"] = total
        final_pos, final_quat = post[n]
        r["reverted_xy_yaw"] = False
        if total["dxy"] > cfg["xy_tol_m"] or total["dyaw_deg"] > cfg["yaw_tol_deg"]:
            msg = (f"{n}: 안착 중 xy {total['dxy'] * 100:.1f}cm, yaw {total['dyaw_deg']:.1f}° 움직임 "
                   f"(z {total['dz'] * 1000:+.1f}mm, tilt {total['dtilt_deg']:.1f}°)")
            if cfg["revert_xy_yaw"] and not is_inside(info):
                final_pos = np.array([pre[n][0][0], pre[n][0][1], final_pos[2]])
                final_quat = pre[n][1]
                r["reverted_xy_yaw"] = True
                msg += " -> xy·자세는 안착 전으로 되돌리고 높이만 반영"
            report["warnings"].append(msg)
            log.warning(f"[physics_settle] {msg}")
        if total["dtilt_deg"] > cfg["tilt_warn_deg"]:
            msg = f"{n}: 안착 중 {total['dtilt_deg']:.0f}° 기울어짐 (넘어지거나 세워졌을 수 있음)"
            report["warnings"].append(msg)
            log.warning(f"[physics_settle] {msg}")
        if not r["stable"]:
            msg = (f"{n}: 안정성 검사 {cfg['check_steps']} 스텝 동안 xy {r['check']['dxy'] * 100:.1f}cm, "
                   f"z {r['check']['dz'] * 1000:+.1f}mm, yaw {r['check']['dyaw_deg']:.1f}° 움직임")
            report["warnings"].append(msg)
            log.warning(f"[physics_settle] {msg}")
        info["tf_from_cam"] = tf_from_cam(final_pos, final_quat, cam_pose)

    # 되돌린 물체가 있으면 렌더링용 씬에도 반영 (스텝하지 않는다)
    for n in names:
        if report["objects"][n]["reverted_xy_yaw"]:
            pos, quat = world_pose(scene_info["objects"][n], cam_pose)
            o = scene.object_registry("name", n)
            o.set_position_orientation(th.tensor(pos, dtype=th.float), th.tensor(quat, dtype=th.float))
            o.keep_still()
    report["inside_check"] = _inside_check(scene, scene_info, names)
    for n, c in report["inside_check"].items():
        if not c["inside"]:
            report["warnings"].append(f"{n}: 안착 후 부모 밖으로 나감 (r_xy {c['r_xy']:.3f} / {c['parent_half']:.3f})")

    for n in names:
        r = report["objects"][n]
        lf = r["lift"]
        log.info(f"[physics_settle] {n:24s} lift {(lf['dz'] if lf else 0) * 1000:+6.1f}mm  "
                 f"settle dxy {r['total']['dxy'] * 1000:5.1f}mm dz {r['total']['dz'] * 1000:+6.1f}mm "
                 f"dyaw {r['total']['dyaw_deg']:4.1f}°  stable={r['stable']}"
                 + ("  (xy·yaw 되돌림)" if r["reverted_xy_yaw"] else ""))
    return scene, to_builtin(report)


def to_builtin(x):
    if isinstance(x, dict):
        return {k: to_builtin(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_builtin(v) for v in x]
    if isinstance(x, (np.ndarray, th.Tensor)):
        return np_(x).tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x
