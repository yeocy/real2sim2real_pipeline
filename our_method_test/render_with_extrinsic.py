"""
render_with_extrinsic.py — 최종 씬을 입력 카메라(camera_info.json)의 extrinsic/intrinsic 으로 찍는다.

GAIA 씬은 사진에서 추정한 자체 좌표계(바닥 z=0, scene_info["cam_pose"])에 있고, camera_info.json 의
extrinsic 은 MuJoCo 월드 좌표다. 두 좌표계를 맞추는 방법을 --align 으로 고른다.

  floor  (기본) 바닥끼리 맞춘다. GAIA 카메라와 GT 카메라의 관계에서 yaw 와 xy 이동만 가져오고
         z 이동과 기울기는 버린다. 씬이 MuJoCo 월드에 놓이므로 GT 카메라로 찍으면 입력 사진과 직접
         비교할 수 있다. GAIA 가 추정한 카메라 높이가 틀렸다면 그 차이가 그대로 드러난다.
  camera 카메라끼리 맞춘다. 물체-카메라 상대 자세를 그대로 두므로 GAIA 카메라로 찍은 것과 같은
         그림이 나온다 (화각만 GT 로 바뀜). 대신 씬이 MuJoCo 바닥에서 떠 있을 수 있다.
  fit    floor 에서 시작해 yaw·xy 를 관측에 맞춰 다시 잡는다. Step 1 마스크 x GT depth 를 GT 카메라로
         역투영해 물체마다 보이는 표면의 MuJoCo 위치(중앙값)를 구하고, GAIA 물체 위치가 거기에
         겹치도록 2D 강체 변환을 최소제곱으로 맞춘다. 입력(depth, extrinsic)만 쓴다.

출력 (기본: <scene_info 폴더>/extrinsic_render/)
  render_<align>.png            GT 카메라로 찍은 씬 (입력과 같은 해상도)
  compare_<align>.png           [입력 | 렌더 | 겹침] 나란히
  scene_info_mujoco_<align>.json MuJoCo 월드로 옮긴 scene_info (view_scene.py 로 열 수 있다)

실행 (acdc env, PYTHONPATH 에 airlab_twin):
  python render_with_extrinsic.py acdc_out_s5_physics/task_scene_generation/scene_info.json \
      [--camera_info inputs_kist_twin_full/camera_info.json] [--align floor|camera]
"""
import os
import json
import argparse

os.environ["OMNIGIBSON_HEADLESS"] = "1"

# og.launch() 뒤에 PIL 을 처음 import 하면 Isaac 번들 PIL 과 섞여 죽는다.
import numpy as np
import torch as th
from PIL import Image
from scipy.spatial.transform import Rotation as R

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


def mat(pos, quat):
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat).as_matrix()
    T[:3, 3] = pos
    return T


def gaia_to_mujoco(T_gaia_cam, T_mj_cam, align):
    """GAIA 월드 -> MuJoCo 월드 변환."""
    X = T_mj_cam @ np.linalg.inv(T_gaia_cam)
    if align == "camera":
        return X
    yaw = R.from_matrix(X[:3, :3]).as_euler("xyz")[2]
    A = np.eye(4)
    A[:3, :3] = R.from_euler("z", yaw).as_matrix()
    # xy 는 카메라 바로 아래 바닥점이 서로 대응하도록 잡는다
    A[:2, 3] = T_mj_cam[:2, 3] - (A[:3, :3] @ T_gaia_cam[:3, 3])[:2]
    return A


def find_seg_dir(scene_info_path):
    """scene_info 위쪽 폴더에서 step_1_output/segmented_objects 를 찾는다."""
    d = os.path.dirname(os.path.abspath(scene_info_path))
    for _ in range(4):
        cand = os.path.join(d, "step_1_output", "segmented_objects")
        if os.path.isdir(cand):
            return cand
        d = os.path.dirname(d)
    return None


def fit_to_observation(A, T_gaia_cam, T_mj_cam, scene_info, cam, depth_path, seg_dir):
    """A(GAIA->MuJoCo) 위에 관측 기준 2D 강체 보정을 곱해 돌려준다."""
    depth = np.load(depth_path).squeeze()
    H, W = depth.shape
    intr = cam["intrinsics"]
    v, u = np.mgrid[0:H, 0:W]
    src, dst, names = [], [], []
    for name, info in scene_info["objects"].items():
        mpath = os.path.join(seg_dir, f"{name}_nonprojected_mask_pruned.png")
        if not os.path.exists(mpath):
            continue        # 그릇 안 내용물 등 Step 1 에 없는 물체는 같이 따라 움직인다
        m = np.array(Image.open(mpath).convert("L").resize((W, H), Image.NEAREST)) > 0
        d = depth[m]
        # 이미지 좌표(x 오른쪽, y 아래) -> OpenGL 카메라 좌표(-z 앞, +y 위)
        p = np.stack([(u[m] - intr["cx"]) / intr["fx"] * d, -(v[m] - intr["cy"]) / intr["fy"] * d, -d, np.ones_like(d)])
        obs = np.median((T_mj_cam @ p)[:3], axis=1)
        src.append((A @ T_gaia_cam @ np.array(info["tf_from_cam"]))[:2, 3])
        dst.append(obs[:2])
        names.append(name)
    src, dst = np.array(src), np.array(dst)
    ms, md = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - ms).T @ (dst - md))
    Rr = Vt.T @ U.T
    if np.linalg.det(Rr) < 0:
        Vt[-1] *= -1
        Rr = Vt.T @ U.T
    t = md - Rr @ ms
    before = np.linalg.norm(src - dst, axis=1)
    after = np.linalg.norm((Rr @ src.T).T + t - dst, axis=1)
    for n, b, a in zip(names, before, after):
        print(f"[fit] {n:22s} {b*100:5.1f} -> {a*100:5.1f} cm")
    print(f"[fit] 보정 yaw {np.degrees(np.arctan2(Rr[1, 0], Rr[0, 0])):+.2f} deg, xy {t.round(3)}, "
          f"평균 오차 {before.mean()*100:.1f} -> {after.mean()*100:.1f} cm")
    F = np.eye(4)
    F[:2, :2] = Rr
    F[:2, 3] = t
    return F @ A


def observed_points(name, seg_dir, depth, cam, T_mj_cam):
    """Step 1 마스크 x GT depth 를 GT 카메라로 역투영한 MuJoCo 월드 점들 (N, 3)."""
    mpath = os.path.join(seg_dir, f"{name}_nonprojected_mask_pruned.png")
    if not os.path.exists(mpath):
        return None
    H, W = depth.shape
    intr = cam["intrinsics"]
    m = np.array(Image.open(mpath).convert("L").resize((W, H), Image.NEAREST)) > 0
    v, u = np.nonzero(m)
    d = depth[m]
    p = np.stack([(u - intr["cx"]) / intr["fx"] * d, -(v - intr["cy"]) / intr["fy"] * d, -d, np.ones_like(d)])
    return (T_mj_cam @ p)[:3].T


def fit_tables(A, T_gaia_cam, T_mj_cam, scene_info, cam, depth_path, seg_dir, categories=("white_table",)):
    """
    책상 윗면을 기준으로 맞춘다.
    관측: 책상 마스크 점 중 윗면(상위 5% 높이 ±2cm)만 골라 위에서 본 최소 외접 사각형(중심, 각도, 크기).
    1) 모든 책상의 각도 차(90도 주기)와 중심 차 평균으로 씬 전체 yaw·xy 를 보정한다.
    2) 그래도 남는 책상별 중심 차는 snap 으로 돌려준다 (책상과 그 위 물체를 같이 옮기는 데 쓴다).
    """
    import cv2
    depth = np.load(depth_path).squeeze()
    obs, gai = {}, {}
    for name, info in scene_info["objects"].items():
        if info["category"] not in categories:
            continue
        P = observed_points(name, seg_dir, depth, cam, T_mj_cam)
        if P is None or len(P) < 100:
            continue
        top = np.percentile(P[:, 2], 95)
        S = P[np.abs(P[:, 2] - top) < 0.02][:, :2].astype(np.float32)
        (cx, cy), (w, h), ang = cv2.minAreaRect(S)
        G = A @ T_gaia_cam @ np.array(info["tf_from_cam"])
        obs[name] = (np.array([cx, cy]), np.radians(ang), (w, h))
        gai[name] = (G[:2, 3].copy(), np.arctan2(G[1, 0], G[0, 0]), info["bbox_extent"][:2])
        print(f"[table] {name}: 관측 중심 {np.round([cx, cy], 3)} 크기 {np.round(sorted([w, h]), 3)} 각도 {ang:.1f} | "
              f"GAIA 중심 {G[:2, 3].round(3)} 크기 {np.round(sorted(info['bbox_extent'][:2]), 3)}")
    assert obs, "책상 관측이 없다"
    # 사각형 각도는 90도 주기로만 의미가 있다
    dyaw = np.mean([((o[1] - g[1] + np.pi / 4) % (np.pi / 2)) - np.pi / 4 for o, g in zip(obs.values(), gai.values())])
    Rz = np.array([[np.cos(dyaw), -np.sin(dyaw)], [np.sin(dyaw), np.cos(dyaw)]])
    gc = np.array([g[0] for g in gai.values()])
    oc = np.array([o[0] for o in obs.values()])
    t = (oc - (Rz @ gc.T).T).mean(0)
    F = np.eye(4)
    F[:2, :2] = Rz
    F[:2, 3] = t
    snap = {}
    for (name, g), o in zip(gai.items(), obs.values()):
        moved = Rz @ g[0] + t
        snap[name] = o[0] - moved
        print(f"[table] {name}: 중심 오차 {np.linalg.norm(o[0]-g[0])*100:.1f} -> 전체 보정 후 "
              f"{np.linalg.norm(snap[name])*100:.1f} cm")
    print(f"[table] 전체 보정 yaw {np.degrees(dyaw):+.2f} deg, xy {t.round(3)}")
    return F @ A, snap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_info")
    ap.add_argument("--camera_info", default=os.path.join(TEST_DIR, "inputs_kist_twin_full", "camera_info.json"))
    ap.add_argument("--rgb", default=None, help="비교할 입력 사진 (기본: camera_info 옆 camera_rgb.png)")
    ap.add_argument("--align", choices=["floor", "camera", "fit", "table"], default="table")
    ap.add_argument("--no_snap", action="store_true", help="table: 책상별 잔여 오차 보정(책상+위 물체 이동)을 끈다")
    ap.add_argument("--depth", default=None, help="GT depth (기본: camera_info 옆 camera_depth.npy). fit 에서 쓴다")
    ap.add_argument("--seg_dir", default=None, help="Step 1 마스크 폴더 (기본: scene_info 위쪽에서 찾음). fit 에서 쓴다")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--usd_pool", default=None,
                    help="USD 를 먼저 불러올 풀 (기본: 통합 풀 kist_twin). 'none' 이면 OG 데이터셋만 쓴다")
    ap.add_argument("--suffix", default="", help="출력 파일 이름 뒤에 붙일 말 (예: whitefloor)")
    ap.add_argument("--lighting", choices=["default", "mujoco"], default="mujoco",
                    help="default: OG 기본 skybox / mujoco: 흰 dome + 카메라 방향광 (MuJoCo headlight 흉내)")
    ap.add_argument("--dome", type=float, default=1500.0)
    ap.add_argument("--key", type=float, default=1500.0)
    ap.add_argument("--calibrate", action="store_true", help="dome/key 세기를 입력 사진에 맞춰 격자 탐색")
    ap.add_argument("--floor_rgb", type=float, nargs=3, default=[0.85, 0.85, 0.85],
                    help="바닥 반사율. 기본은 MuJoCo floor_tex (0.85 회색). OG 기본은 흰색이라 물체와 구분이 안 된다")
    ap.add_argument("--white_floor", action="store_true",
                    help="찍은 뒤 바닥 픽셀(semantic 'floors')만 순백으로 칠한다. 물체 명암·투명도는 그대로 둔다")
    ap.add_argument("--floor_emissive", type=float, default=0.0,
                    help="0 보다 크면 바닥을 floor_rgb 색으로 스스로 빛나게 한다 (조명과 무관하게 완전한 흰 배경 등)")
    args = ap.parse_args()

    scene_info = json.load(open(args.scene_info))
    cam = json.load(open(args.camera_info))
    rgb_path = args.rgb or os.path.join(os.path.dirname(args.camera_info), "camera_rgb.png")
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.scene_info)), "extrinsic_render")
    os.makedirs(out_dir, exist_ok=True)

    T_gaia_cam = mat(*scene_info["cam_pose"])
    T_mj_cam = mat(cam["camera"]["position"], cam["camera"]["orientation"])
    A = gaia_to_mujoco(T_gaia_cam, T_mj_cam, "floor" if args.align in ("fit", "table") else args.align)
    snap = {}
    if args.align in ("fit", "table"):
        depth_path = args.depth or os.path.join(os.path.dirname(args.camera_info), "camera_depth.npy")
        seg_dir = args.seg_dir or find_seg_dir(args.scene_info)
        assert seg_dir and os.path.exists(depth_path), f"필요한 depth({depth_path}) / 마스크({seg_dir}) 가 없다"
        if args.align == "fit":
            A = fit_to_observation(A, T_gaia_cam, T_mj_cam, scene_info, cam, depth_path, seg_dir)
        else:
            A, snap = fit_tables(A, T_gaia_cam, T_mj_cam, scene_info, cam, depth_path, seg_dir)
            if args.no_snap:
                snap = {}
    print(f"[extrinsic] align={args.align}: yaw {np.degrees(R.from_matrix(A[:3, :3]).as_euler('xyz')).round(2)} deg, "
          f"trans {A[:3, 3].round(3)}")
    print(f"[extrinsic] 카메라 높이: GAIA {T_gaia_cam[2, 3]:.3f} m, GT {T_mj_cam[2, 3]:.3f} m")

    # MuJoCo 월드로 옮긴 scene_info: 카메라는 GT, 물체는 A 로 옮긴 월드 자세를 GT 카메라 기준으로 다시 쓴다
    new_info = json.loads(json.dumps(scene_info))
    new_info["cam_pose"] = [list(cam["camera"]["position"]), list(cam["camera"]["orientation"])]
    T_mj_cam_inv = np.linalg.inv(T_mj_cam)
    # 책상별 잔여 보정: 책상과 그 위(받침 관계를 따라 올라간) 물체를 모두 같이 옮긴다
    offset = {}
    if snap:
        from our_method.utils.physics_settle import supports_of
        sup = supports_of(scene_info)

        def root_table(n):
            seen = set()
            while n is not None and n not in snap and n not in seen:
                seen.add(n)
                n = sup.get(n)
            return n if n in snap else None
        for name in new_info["objects"]:
            r = root_table(name)
            if r is not None:
                offset[name] = snap[r]
    for name, info in new_info["objects"].items():
        W = A @ T_gaia_cam @ np.array(info["tf_from_cam"], dtype=np.float64)
        if name in offset:
            W[:2, 3] += offset[name]
        info["tf_from_cam"] = (T_mj_cam_inv @ W).tolist()
    if offset:
        print(f"[table] 책상별 보정에 같이 옮긴 물체 {len(offset)}개")
    info_path = os.path.join(out_dir, f"scene_info_mujoco_{args.align}.json")
    json.dump(new_info, open(info_path, "w"), indent=2)

    import omnigibson as og
    from omnigibson.objects import LightObject
    from our_method.utils.physics_settle import load_scene
    from our_method.utils.asset_pool import use_pool_usd, DEFAULT_POOL_ROOT  # launch 전에 import (PIL 충돌)

    og.launch()
    if (args.usd_pool or "").lower() != "none":
        use_pool_usd(args.usd_pool or DEFAULT_POOL_ROOT)
    scene = load_scene(new_info, "static", [])
    intr = cam["intrinsics"]
    W, H = int(intr["image_width"]), int(intr["image_height"])
    vc = og.sim.viewer_camera
    vc.image_width = W
    vc.image_height = H
    # 핀홀: fx[px] = focal[mm] * width[px] / horizontal_aperture[mm]
    vc.focal_length = float(intr["fx"]) * vc.horizontal_aperture / float(W)
    cam_pos = th.tensor(cam["camera"]["position"], dtype=th.float)
    cam_quat = th.tensor(cam["camera"]["orientation"], dtype=th.float)
    vc.set_position_orientation(cam_pos, cam_quat)
    if args.white_floor:
        vc.add_modality("seg_semantic")

    key = None
    if args.lighting == "mujoco":
        key = setup_mujoco_lighting(og, scene, LightObject, cam_pos, cam_quat)
        set_floor_color(og, args.floor_rgb, args.floor_emissive)
    og.sim.play()
    og.sim.step()

    def shot(n=30):
        for _ in range(n):
            og.sim.render()
        im = vc.get_obs()[0]["rgb"][:, :, :3].cpu().numpy()
        return im if im.dtype == np.uint8 else (im * 255).astype(np.uint8)

    ref = np.array(Image.open(rgb_path).convert("RGB").resize((W, H))) if os.path.exists(rgb_path) else None
    tag = f"{args.align}_{args.lighting}" + (f"_{args.suffix}" if args.suffix else "")

    if args.lighting == "mujoco":
        dome, key_int = args.dome, args.key
        if args.calibrate and ref is not None:
            regions = build_regions(args.seg_dir or find_seg_dir(args.scene_info), W, H)
            best = None
            for d in CAL_DOME:
                for k in CAL_KEY:
                    set_intensity(og, key, d, k)
                    shot(40)        # 세기를 바꾼 직후 프레임은 이전 조명이 섞여 있다. 충분히 수렴시킨 뒤 잰다
                    err = region_error(shot(20), ref, regions)
                    print(f"[light] dome {d:7.0f} key {k:7.0f} -> 영역 평균색 오차 {err:6.2f}")
                    if best is None or err < best[0]:
                        best = (err, d, k)
            _, dome, key_int = best
            print(f"[light] 선택: dome {dome:.0f}, key {key_int:.0f} (오차 {best[0]:.2f})")
            json.dump({"dome": dome, "key": key_int, "error": best[0]},
                      open(os.path.join(out_dir, f"lighting_{tag}.json"), "w"), indent=2)
        set_intensity(og, key, dome, key_int)

    img = shot()
    if args.white_floor:
        obs, info = vc.get_obs()
        seg = obs["seg_semantic"].cpu().numpy()
        names = info.get("seg_semantic", {})
        floor_ids = [int(i) for i, n in names.items() if n in ("floors", "background")]
        floor = np.isin(seg, floor_ids)
        img = img.copy()
        img[floor] = 255
        print(f"[extrinsic] 바닥 픽셀 {floor.mean()*100:.1f}% 를 흰색으로 칠함 (classes {[names[i] for i in names if int(i) in floor_ids]})")
    render_path = os.path.join(out_dir, f"render_{tag}.png")
    Image.fromarray(img).save(render_path)
    if ref is not None:
        blend = (0.5 * ref + 0.5 * img).astype(np.uint8)
        Image.fromarray(np.concatenate([ref, img, blend], axis=1)).save(os.path.join(out_dir, f"compare_{tag}.png"))
    print(f"[extrinsic] saved {render_path}, {info_path}")
    og.shutdown()


# MuJoCo 씬은 따로 둔 조명 없이 headlight(카메라에서 나오는 방향광 + ambient)만 쓴다.
# OG 의 기본 skybox(하늘 텍스처, 따뜻한 색 dome)는 흰 물체에 푸른/노란 기를 입히므로
# 텍스처 없는 흰 dome(= ambient) + 카메라 방향 distant light(= headlight)로 바꾼다.
CAL_DOME = [0, 100, 200, 300, 450]
CAL_KEY = [1000, 1250, 1500, 1750, 2000]


def setup_mujoco_lighting(og, scene, LightObject, cam_pos, cam_quat):
    sky = getattr(og.sim, "_skybox", None)
    if sky is not None:
        try:
            sky.texture_file_path = ""
        except Exception:
            sky._light_link.set_attribute("inputs:texture:file", "")
        sky.color = (1.0, 1.0, 1.0)
    with og.sim.stopped():
        key = LightObject(name="headlight", light_type="Distant", intensity=0.0)
        scene.add_object(key)
    # UsdLux DistantLight 은 자기 -Z 로 비춘다. OpenGL 카메라도 -Z 를 보므로 같은 자세를 주면 된다.
    key.set_position_orientation(cam_pos, cam_quat)
    # MuJoCo headlight 는 그림자를 만들지 않는다. 평행광 그림자는 화면 가장자리(시선과 최대 fovy/2 차이)에서
    # 책상 아래 검은 쐐기로 보이므로 끈다.
    from pxr import UsdLux
    light_prim = og.sim.stage.GetPrimAtPath(f"{key.prim_path}/base_link/light")
    UsdLux.ShadowAPI.Apply(light_prim).CreateShadowEnableAttr().Set(False)
    return key


def set_floor_color(og, rgb, emissive=0.0):
    """Isaac GroundPlane 에 묶인 재질의 diffuse 색을 바꾼다 (PreviewSurface / OmniPBR 둘 다)."""
    from pxr import Gf, Sdf, Usd, UsdShade
    stage = og.sim.stage
    root = stage.GetPrimAtPath("/World/ground_plane")
    n = 0
    for prim in Usd.PrimRange(root):
        mat = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        if not mat:
            continue
        for sh in Usd.PrimRange(mat.GetPrim()):
            for attr in ("inputs:diffuseColor", "inputs:diffuse_color_constant"):
                a = sh.GetAttribute(attr)
                if not a:
                    a = sh.CreateAttribute(attr, Sdf.ValueTypeNames.Color3f) if sh.IsA(UsdShade.Shader) and attr == "inputs:diffuse_color_constant" and "OmniPBR" in str(sh.GetAttribute("info:mdl:sourceAsset").Get()) else None
                if a:
                    a.Set(Gf.Vec3f(*map(float, rgb)))
                    n += 1
            if emissive > 0 and sh.IsA(UsdShade.Shader):
                src = str(sh.GetAttribute("info:mdl:sourceAsset").Get())
                if "OmniPBR" in src:
                    sh.CreateAttribute("inputs:enable_emission", Sdf.ValueTypeNames.Bool).Set(True)
                    sh.CreateAttribute("inputs:emissive_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, rgb)))
                    sh.CreateAttribute("inputs:emissive_intensity", Sdf.ValueTypeNames.Float).Set(float(emissive))
                else:
                    sh.CreateAttribute("inputs:emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
                        Gf.Vec3f(*[float(c) * float(emissive) for c in rgb]))
    print(f"[light] 바닥 색 {rgb}, 발광 {emissive} (재질 속성 {n}개)")


def set_intensity(og, key, dome, key_int):
    if getattr(og.sim, "_skybox", None) is not None:
        og.sim._skybox.intensity = float(dome)
    if key is not None:
        key.intensity = float(key_int)


def build_regions(seg_dir, W, H):
    """입력 사진 기준 비교 영역: 바닥 + 물체별 마스크(경계 어긋남을 줄이려고 안쪽으로 깎는다)."""
    from scipy.ndimage import binary_erosion
    regions = []
    if seg_dir is None:
        return regions
    union = np.zeros((H, W), dtype=bool)
    for f in sorted(os.listdir(seg_dir)):
        if f.endswith("_nonprojected_mask_pruned.png"):
            path = os.path.join(seg_dir, f)
        elif f == "floor":
            path = os.path.join(seg_dir, "floor", "floor_mask.png")
        else:
            continue
        if not os.path.exists(path):
            continue
        m = np.array(Image.open(path).convert("L").resize((W, H), Image.NEAREST)) > 0
        union |= m
        m = binary_erosion(m, iterations=6)
        if m.sum() > 200:
            regions.append(m)
    # 어느 마스크에도 안 든 배경(대부분 바닥)도 한 영역으로 넣는다
    bg = binary_erosion(~union, iterations=6)
    if bg.sum() > 200:
        regions.append(bg)
    return regions


def region_error(img, ref, regions):
    """영역별 평균색 차이(L2) 평균 + 전체 픽셀 L1.
    영역 평균만 보면 책상 다리 안쪽처럼 작은 면이 새까매져도 잡지 못한다. 위치가 맞춰진 뒤에는
    픽셀 단위 L1 이 명암(그늘진 면)까지 반영한다."""
    l1 = float(np.abs(img.astype(float) - ref).mean())
    if not regions:
        return l1
    errs = [np.linalg.norm(img[m].astype(float).mean(0) - ref[m].astype(float).mean(0)) for m in regions]
    return float(np.mean(errs)) + l1


if __name__ == "__main__":
    main()
