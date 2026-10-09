"""
sam3d_asset_builder.py — SAM3D 메시(.glb)로 GAIA retrieval 풀을 만든다.

입력 사진(+깊이, 카메라)에서 SAM3D 로 뽑은 물체 메시를 실치수로 맞춰 OG USD 로 만들고,
retrieval 용 뷰 렌더까지 해서 바로 config 의 asset_pool 로 쓸 수 있는 풀을 만든다.

  [sam3d]  입력 사진 + 프롬프트 -> 원격 SAM3 마스크 + SAM 3D Objects 메시 (Kist/Kist/SAM3D/sam3d_remote.sh)
           결과(GLB)가 이미 있으면 건너뛴다 (--force-sam3d 로 다시 생성)
    -> [mesh]   GLB -> 눕히기(y-up -> z-up) -> 실치수 스케일 -> 감축 -> UV 전개 + 텍스처 굽기 -> npz
    -> [align]  에셋 크기를 depth 에 맞춘다: yaw 는 입력 카메라 render-and-compare, 크기는 depth 점군 3D 정합으로
                메시 x/y/z 축별 배율을 정해 npz 메시에 굽는다 (our_method/utils/sam3d_pose_alignment.py).
                sam3d_assets.json 의 행마다 detection 이름과 registration.pose_cam 을 남기고, Step 3 이 그 pose 로
                바로 놓는다. 비교 그림 <pool>/alignment.png. 크기가 바뀌므로 usd, render 를 그 뒤에 돈다
    -> [usd]    npz -> OG DatasetObject USD (visual 텍스처 + convex hull 충돌) -> 암호화
                <pool>/og_dataset/objects/<cat>/<model>/usd/<model>.encrypted.usd
    -> [render] 검은 배경, 앙각 --elevation, 1280x720, yaw 100장 + snapshot
                <pool>/objects/<cat>/model/<model>/<model>_<0..99>.png
                <pool>/objects/<cat>/snapshot/<cat>_<model>.png
    -> [link]   link_categories 를 기본 assets/ 에서 심볼릭 링크 (테이블 등 SAM3D 로 안 만드는 것)
    -> [overview] <pool>/overview.png: 물체별 정보 | SAM3D 입력 크롭(마스크) | 뷰 4장 소개 이미지

실치수: 마스크 영역의 깊이를 월드로 역투영해서 잰다.
  height   = 마스크 점 최고 z - 마스크 바로 바깥 링의 z 중앙값(받침 면). 가림에 강하다.
  diameter = 마스크 점 xy 의 최대 거리. 납작한 물체(접시)는 위가 다 보여서 이걸 쓴다.
  size_from 기본값: 납작한 물체(메시 높이/폭 < 0.3)는 diameter 로 균일 스케일,
  나머지는 both = z 는 height, xy 는 diameter 로 따로 맞춘다. SAM3D 메시는 비율이 틀어지는
  경우가 있다 (LIBERO 머그: 손잡이가 길어 높이로만 맞추면 폭이 30% 크다).
메시에 실치수를 구워 넣으므로 Step 3 는 이 모델들의 스케일을 바꾸지 않는다. 풀 루트의
sam3d_assets.json(scale_baked: true)을 use_pool_usd 가 읽어 등록하면 scene_utils.align_model_pose 가
자동으로 건너뛴다 (GAIA_NO_FIT_SCALE 불필요). 테이블 등 link_categories 는 기존대로 점군에 맞춘다.

텍스처 (기본): SAM3D GLB 는 UV 없이 정점색만 있다. 감축(40k 면)하면 색 디테일이 같이 사라지고,
  정점색(sRGB)을 선형 diffuse 로 넘기면 색이 바래 보인다. 그래서 감축 메시를 xatlas 로 UV 전개하고,
  텍셀마다 원본 해상도 메시(수십만 정점)의 가장 가까운 정점들의 색을 보간해 PNG 로 굽는다.
  PNG 는 sRGB 로 읽히므로(build_og_usd._make_material) 색공간도 맞는다.
  --no-texture 면 예전처럼 감축 메시의 정점색을 쓴다.

yaw 규약: 인덱스 i -> z_angle = i*2pi/100 - pi (dataset_utils.extract_info_from_model_snapshot),
카메라는 -Y 쪽 (Kist/Kist/og_export/render_asset_views.py 와 같다).

spec (yaml):
  pool: asset_pools_local/libero10_04          # 만들 풀 (TEST_DIR 기준 상대경로 가능)
  inputs: inputs_libero_10/04_...              # camera_depth.npy, camera_info.json
  sam3d_dir: sam3d_libero/04                   # mesh_<stem>_<prompt>_<i>.glb, sam3/<stem>/<prompt>.npy
  image_stem: libero04                         # SAM3D 입력 이미지 = <sam3d_dir>/<image_stem>.png
  sam3d:                                       # [sam3d] 단계 설정
    image: /path/to/source.png                 #   <sam3d_dir>/<image_stem>.png 로 복사해서 쓴다
    prompts: "mug,plate"                       #   SAM3 텍스트 프롬프트 (쉼표 구분)
    gpus: "6,7"                                #   원격 서버의 빈 GPU (mesh 디코더 때문에 2개)
    from_masks: false                          #   true: SAM3 를 건너뛰고 <sam3d_dir>/sam3/<stem>/<prompt>.npy
                                               #   마스크로 SAM3D 만 돌린다 (파이프라인 Step 1.5 가 Step 1
                                               #   마스크를 넣어 쓴다, sam3d_asset_generation.py)
  title: "put the white mug on ..."            # overview.png 제목 줄 (선택)
  elevation: 40
  link_categories: [coffee_table, ...]
  assets:
    - {instance: mug_1, category: mug, model: lwhitemug}
    - {instance: plate_0, category: plate, model: lplate, size_from: diameter, mass: 0.4}

실행 (acdc env, PYTHONPATH 에 airlab_twin):
  python sam3d_asset_builder.py --spec sam3d_libero/04/spec.yaml         # SAM3D 생성부터 끝까지
  python sam3d_asset_builder.py --spec ... --stages sam3d --sam3d-list   # 검출 인스턴스만 확인
  (assets: 의 instance 이름은 --sam3d-list 결과의 <prompt>#<i> 를 <prompt>_<i> 로 적는다)
  python sam3d_asset_builder.py --spec ... --stages mesh          # 치수만 확인
  python sam3d_asset_builder.py --spec ... --stages render --n-views 8   # 빠른 렌더 확인
"""
import argparse
import json
import math
import os
import sys

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")

# og.launch() 뒤에 PIL 을 처음 import 하면 Isaac 번들 PIL 과 섞여 죽는다. 먼저 import 한다.
import cv2
import numpy as np
import yaml
from PIL import Image
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SAM3D_REMOTE = "/home/yeocy/robotics/Kist/Kist/SAM3D/sam3d_remote.sh"
SAM3D_FROM_MASK = "/home/yeocy/robotics/Kist/Kist/SAM3D/sam3d_from_mask.sh"
sys.path.insert(0, os.path.dirname(TEST_DIR))
# our_method 가 torchvision -> PIL 을 끌어오므로 이것도 og.launch() 전에 import 한다.
from our_method.utils.asset_pool import use_pool_usd  # noqa: E402
# USD 작성 함수(build_usd)는 KIST 변환 스크립트의 것을 그대로 쓴다 (읽기만 한다).
KIST_OG_EXPORT = "/home/yeocy/robotics/Kist/Kist/og_export"

N_VIEWS = 100            # dataset_utils 가 가정하는 각도 수
W, H = 1280, 720         # 기존 assets/ 스냅샷 규격
SNAPSHOT_IDX = 62        # yaw 43.2도 3/4 뷰 (render_asset_views.py 와 같다)
TARGET_TRIS = 40000
FLAT_RATIO = 0.3
TEX_SIZE = 2048
TEX_K = 4                # 텍셀 색 = 원본 메시 최근접 정점 K 개의 거리 가중 평균


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(TEST_DIR, p)


# ----------------------------------------------------------------------- 실치수
def backproject(depth, info, mask):
    """마스크 픽셀 -> 월드 점. camera_info 는 opengl(-z 전방, +y 위), xyzw."""
    K = np.array(info["intrinsics"]["matrix"])
    Rm = Rotation.from_quat(info["camera"]["orientation"]).as_matrix()
    p = np.array(info["camera"]["position"])
    v, u = np.nonzero(mask)
    z = depth[v, u]
    pc = np.stack([(u - K[0, 2]) * z / K[0, 0], -(v - K[1, 2]) * z / K[1, 1], -z], axis=1)
    return pc @ Rm.T + p


def xy_diameter(pts):
    xy = pts[:, :2]
    if len(xy) > 3:
        xy = xy[ConvexHull(xy).vertices]
    d = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    return float(d.max())


def measure(depth, info, mask):
    """마스크 물체의 (height, diameter) [m]."""
    mask = cv2.resize(mask.astype(np.uint8), depth.shape[::-1], interpolation=cv2.INTER_NEAREST)
    k = np.ones((3, 3), np.uint8)
    inner = cv2.erode(mask, k, iterations=2).astype(bool)        # 경계의 배경 깊이 섞임 제거
    ring = (cv2.dilate(mask, k, iterations=8) - cv2.dilate(mask, k, iterations=3)).astype(bool)
    pts = backproject(depth, info, inner)
    support_z = float(np.median(backproject(depth, info, ring)[:, 2]))
    return float(np.percentile(pts[:, 2], 99.5) - support_z), xy_diameter(pts), support_z


def register_mesh(V, F, depth, info, mask, s_init, n_samples=12000, yaw_steps=36):
    """SAM3D 메시를 관측 깊이 점군에 정합한다: yaw, xy 스케일, z 스케일, xy 위치 (바닥은 받침면에 붙인다).

    높이/지름 두 숫자로만 맞추면 (1) 손잡이처럼 가려지거나 SAM3D 가 길게 만든 부분 때문에 몸통 폭이
    틀어지고 (2) yaw 를 모른다. 여기서는 보이는 표면 전체를 쓴다.
      cost = (관측점 -> 카메라에서 보이는 메시 표면) 거리  +  실루엣 불일치(마스크 밖으로 나간 비율 + 못 덮은 비율)

    Args:
        V (N,3): z-up, xy 중심, 바닥 z=0 인 메시 정점 (스케일 전)
        s_init (3,): 높이/지름으로 구한 초기 스케일

    Returns:
        dict: yaw, scale(3,), t(3,) (메시 바닥 중심의 월드 위치), cost, support_z, n_points
    """
    import trimesh
    from scipy.optimize import minimize
    from scipy.spatial import cKDTree

    H, W = depth.shape
    m = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    k = np.ones((3, 3), np.uint8)
    inner = cv2.erode(m, k, iterations=2).astype(bool)
    ring = (cv2.dilate(m, k, iterations=8) - cv2.dilate(m, k, iterations=3)).astype(bool)
    P = backproject(depth, info, inner)
    support_z = float(np.median(backproject(depth, info, ring)[:, 2]))
    if len(P) > 4000:
        P = P[np.random.default_rng(0).choice(len(P), 4000, replace=False)]
    mask_d = cv2.dilate(m, k, iterations=2).astype(bool)
    n_mask = int(m.sum())

    mesh = trimesh.Trimesh(V, F, process=False)
    S, fi = trimesh.sample.sample_surface(mesh, n_samples, seed=0)
    N = mesh.face_normals[fi]
    Kmat = np.array(info["intrinsics"]["matrix"])
    Rwc = Rotation.from_quat(info["camera"]["orientation"]).as_matrix()
    pc = np.array(info["camera"]["position"])

    def transform(x):
        yaw, sxy, sz, tx, ty = x
        R = Rotation.from_euler("z", yaw).as_matrix()
        s = np.array([sxy, sxy, sz])
        X = (S * s) @ R.T + np.array([tx, ty, support_z])
        n = (N / s) @ R.T
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        return X, n

    def cost(x):
        X, n = transform(x)
        vis = ((X - pc) * n).sum(1) < 0                       # 카메라를 향한 면
        Xv = X[vis] if vis.sum() > 50 else X
        d, _ = cKDTree(Xv).query(P)
        geo = np.mean(np.minimum(d, 0.03)) / 0.003            # 관측점이 표면 위에 있어야 한다 (3mm 단위)
        # 실루엣: 보이는 표면을 이미지로 투영
        Xc = (Xv - pc) @ Rwc                                  # world -> cam (opengl)
        z = -Xc[:, 2]
        ok = z > 1e-3
        u = (Kmat[0, 0] * Xc[ok, 0] / z[ok] + Kmat[0, 2]).astype(int)
        v = (-Kmat[1, 1] * Xc[ok, 1] / z[ok] + Kmat[1, 2]).astype(int)
        inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v = u[inb], v[inb]
        outside = 1.0 - (mask_d[v, u].mean() if len(u) else 0.0)
        cov = np.zeros((H, W), np.uint8)
        cov[v, u] = 1
        cov = cv2.dilate(cov, k, iterations=2).astype(bool)
        uncovered = 1.0 - (cov & m.astype(bool)).sum() / max(n_mask, 1)
        return geo + 4.0 * outside + 4.0 * uncovered

    c0 = P[:, :2].mean(0)
    starts = []
    for yaw in np.linspace(-np.pi, np.pi, yaw_steps, endpoint=False):
        x0 = np.array([yaw, s_init[0], s_init[2], c0[0], c0[1]])
        starts.append((cost(x0), x0))
    starts.sort(key=lambda t: t[0])
    best = None
    for _, x0 in starts[:4]:
        r = minimize(cost, x0, method="Powell",
                     options={"maxiter": 3000, "xtol": 1e-4, "ftol": 1e-4})
        if best is None or r.fun < best.fun:
            best = r
    yaw, sxy, sz, tx, ty = best.x
    return {"yaw": float((yaw + np.pi) % (2 * np.pi) - np.pi), "scale": [float(sxy), float(sxy), float(sz)],
            "t": [float(tx), float(ty), support_z], "cost": float(best.fun), "support_z": support_z,
            "n_points": int(len(P))}


def pose_mat(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ----------------------------------------------------------------------- sam3d
def stage_sam3d(spec, assets, force=False, list_only=False):
    """원격 SAM3D 로 GLB 를 만든다. 필요한 GLB 가 다 있으면 건너뛴다."""
    import shutil
    import subprocess
    cfg = spec.get("sam3d") or {}
    stem = spec["image_stem"]
    img = os.path.join(spec["sam3d_dir"], f"{stem}.png")
    os.makedirs(spec["sam3d_dir"], exist_ok=True)
    if cfg.get("image") and os.path.abspath(_abs(cfg["image"])) != os.path.abspath(img):
        shutil.copy(_abs(cfg["image"]), img)
    if not os.path.exists(img):
        sys.exit(f"[sam3d] 입력 이미지가 없다: {img} (spec 의 sam3d.image 를 적을 것)")
    glbs = [os.path.join(spec["sam3d_dir"], f"mesh_{stem}_{a['instance']}.glb") for a in assets]
    if not list_only and not force and all(os.path.exists(g) for g in glbs):
        print(f"[sam3d] GLB {len(glbs)}개가 이미 있다 -> 건너뜀 (--force-sam3d 로 다시 생성)")
        return
    env = dict(os.environ)
    if cfg.get("gpus"):
        env["SAM3_GPUS"] = str(cfg["gpus"])
    if cfg.get("from_masks"):
        # 이미 있는 마스크(Step 1)로 물체마다 SAM3D 만 돌린다. 출력: mesh_<stem>_<prompt>_0.glb
        for a, g in zip(assets, glbs):
            if os.path.exists(g) and not force:
                continue
            prompt = a["instance"].rsplit("_", 1)[0]
            mask = os.path.join(spec["sam3d_dir"], "sam3", stem, f"{prompt}.npy")
            cmd = [SAM3D_FROM_MASK, img, mask, spec["sam3d_dir"], prompt]
            print(f"[sam3d] {' '.join(cmd)}  (SAM3_GPUS={env.get('SAM3_GPUS', '-')})")
            subprocess.run(cmd, env=env, check=True)
        missing = [g for g in glbs if not os.path.exists(g)]
        if missing:
            sys.exit(f"[sam3d] 생성되지 않은 GLB: {missing}")
        return
    prompts = cfg.get("prompts") or ",".join(sorted({a["instance"].rsplit("_", 1)[0] for a in assets}))
    cmd = [SAM3D_REMOTE, img, spec["sam3d_dir"], prompts]
    if list_only:
        cmd.append("--list")
    elif not force:     # 없는 인스턴스만 만든다
        # instance 이름 <prompt>_<i> -> 드라이버 지정자 <prompt>#<i>
        miss = ["#".join(a["instance"].rsplit("_", 1)) for a, g in zip(assets, glbs) if not os.path.exists(g)]
        cmd += ["--only", ",".join(miss)]
    print(f"[sam3d] {' '.join(cmd)}  (SAM3_GPUS={env.get('SAM3_GPUS', '-')})")
    subprocess.run(cmd, env=env, check=True)
    if not list_only:
        missing = [g for g in glbs if not os.path.exists(g)]
        if missing:
            sys.exit(f"[sam3d] 생성되지 않은 GLB: {missing}")


# ------------------------------------------------------------------------ mesh
def load_glb(path):
    """(감축 V, F, 감축 정점색, 원본 V, 원본 정점색). 정점은 z-up 으로 돌려서 준다."""
    import trimesh
    sc = trimesh.load(path)
    m = trimesh.util.concatenate(list(sc.geometry.values())) if hasattr(sc, "geometry") else sc
    rot = Rotation.from_euler("x", 90, degrees=True).as_matrix().T      # SAM3D y-up -> z-up
    colors = None
    vc = getattr(m.visual, "vertex_colors", None)
    if vc is not None and len(vc) == len(m.vertices):
        colors = np.asarray(vc)[:, :3].astype(np.float64) / 255.0
    V_full, C_full = np.asarray(m.vertices, dtype=np.float64) @ rot, colors
    if len(m.faces) > TARGET_TRIS:
        simp = m.simplify_quadric_decimation(face_count=TARGET_TRIS)
        if colors is not None:
            _, idx = trimesh.proximity.ProximityQuery(m).vertex(simp.vertices)
            colors = colors[idx]
        m = simp
    V = np.asarray(m.vertices, dtype=np.float64) @ rot
    return V, np.asarray(m.faces, dtype=np.int64), colors, V_full, C_full


def bake_texture(V, F, V_full, C_full, size=TEX_SIZE):
    """감축 메시를 UV 전개하고 원본 정점색을 텍스처로 굽는다.

    returns (V', F', UV, tex uint8 HxWx3). xatlas 가 솔기에서 정점을 쪼개므로 V' 가 늘어난다.
    UV 는 USD st 규약(v 위쪽)이고, 이미지 행 0 이 v=1 이다.
    """
    import xatlas
    from scipy.spatial import cKDTree

    vmap, Fn, UV = xatlas.parametrize(V.astype(np.float32), F.astype(np.uint32))
    Vn, Fn, UV = V[vmap], Fn.astype(np.int64), UV.astype(np.float64)

    # 삼각형 id 맵: 텍셀 중심이 어느 면에 속하는지 (cv2 로 면마다 채운다)
    px = np.stack([UV[:, 0] * size, (1.0 - UV[:, 1]) * size], axis=1)
    fid = np.full((size, size), -1, np.int32)
    for i, tri in enumerate(Fn):
        cv2.fillConvexPoly(fid, np.round(px[tri] * 16).astype(np.int32), int(i), lineType=cv2.LINE_8, shift=4)
    ys, xs = np.nonzero(fid >= 0)
    f = fid[ys, xs]
    # 텍셀 중심의 무게중심 좌표 -> 3D 점
    a, b, c = px[Fn[f, 0]], px[Fn[f, 1]], px[Fn[f, 2]]
    q = np.stack([xs + 0.5, ys + 0.5], axis=1)
    v0, v1, v2 = b - a, c - a, q - a
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    den[np.abs(den) < 1e-12] = 1e-12
    w1 = np.clip((d11 * d20 - d01 * d21) / den, 0, 1)
    w2 = np.clip((d00 * d21 - d01 * d20) / den, 0, 1)
    w0 = np.clip(1 - w1 - w2, 0, 1)
    P = w0[:, None] * Vn[Fn[f, 0]] + w1[:, None] * Vn[Fn[f, 1]] + w2[:, None] * Vn[Fn[f, 2]]

    # 원본 메시 최근접 정점 K 개 거리 가중 평균
    dist, idx = cKDTree(V_full).query(P, k=TEX_K)
    w = 1.0 / np.maximum(dist, 1e-6)
    col = (C_full[idx] * w[..., None]).sum(1) / w.sum(1, keepdims=True)

    tex = np.zeros((size, size, 3), np.uint8)
    tex[ys, xs] = np.clip(col * 255 + 0.5, 0, 255).astype(np.uint8)
    # 솔기 번짐 방지: 빈 텍셀을 주변 색으로 몇 픽셀 채운다 (밉맵/보간 시 검은 테두리 방지)
    filled = (fid >= 0).astype(np.uint8)
    for _ in range(8):
        dil = cv2.dilate(tex, np.ones((3, 3), np.uint8))
        grow = cv2.dilate(filled, np.ones((3, 3), np.uint8)) & (1 - filled)
        tex[grow > 0] = dil[grow > 0]
        filled |= grow
    return Vn, Fn, UV, tex


def stage_mesh(spec, a, texture=True):
    import trimesh
    stem = spec["image_stem"]
    prompt, idx = a["instance"].rsplit("_", 1)
    V, F, colors, V_full, C_full = load_glb(os.path.join(spec["sam3d_dir"], f"mesh_{stem}_{a['instance']}.glb"))
    mask = np.load(os.path.join(spec["sam3d_dir"], "sam3", stem, f"{prompt}.npy"))[int(idx)]
    depth = np.load(os.path.join(spec["inputs"], "camera_depth.npy"))[..., 0]
    info = json.load(open(os.path.join(spec["inputs"], "camera_info.json")))

    h_w, d_w, _ = measure(depth, info, mask)
    # 정합 전에 xy 중심, 바닥 z=0 으로 맞춘다 (정합/텍스처/USD 가 모두 이 프레임을 쓴다)
    off0 = np.array([*(V[:, :2].max(0) + V[:, :2].min(0)) / 2, V[:, 2].min()])
    V, V_full = V - off0, V_full - off0
    h_m, d_m = float(np.ptp(V[:, 2])), xy_diameter(V)
    mode = a.get("size_from") or ("diameter" if h_m / d_m < FLAT_RATIO else "both")
    reg = None
    if "size" in a:                                   # 실측값을 직접 줄 때 (x, y, z)
        s = np.array(a["size"]) / np.ptp(V, axis=0)
        mode = "explicit"
    else:
        s_init = np.array([d_w / d_m, d_w / d_m, h_w / h_m]) if mode == "both" else \
            np.full(3, h_w / h_m if mode == "height" else d_w / d_m)
        if a.get("register", False):
            # [실험 중, 기본 꺼짐] 보이는 표면 전체로 yaw, xy/z 스케일, 위치를 맞춘다.
            # 아직 Step 3 에 연결되지 않았고 납작한 물체의 z 스케일이 불안정하다.
            # 설계/현황: artifacts/05_SAM3D_ASSET_ALIGNMENT.md
            reg = register_mesh(V, F, depth, info, mask, s_init)
            s = np.array(reg["scale"])
            mode = "register"
        else:
            s = s_init
    # 원본 메시도 같은 변환을 거쳐야 텍스처를 구울 때 위치가 맞는다
    V, V_full = V * s, V_full * s
    hull = trimesh.Trimesh(V, F).convex_hull
    out = os.path.join(spec["pool"], "_build", f"{a['model']}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    uvs, tex_path = np.zeros((0, 2)), ""
    if texture and C_full is not None:
        V, F, uvs, tex = bake_texture(V, F, V_full, C_full)
        tex_path = os.path.join(spec["pool"], "_build", f"{a['model']}_albedo.png")
        cv2.imwrite(tex_path, tex[:, :, ::-1])
        colors = None
    np.savez(out, verts=V, faces=F, colors=colors if colors is not None else np.zeros((0, 3)),
             uvs=uvs, tex=tex_path, col_verts=hull.vertices, col_faces=hull.faces)
    ext = np.ptp(V, axis=0)
    row = {"category": a["category"], "model": a["model"], "size_from": mode,
           "observed_height": h_w, "observed_diameter": d_w, "extent": ext.tolist(),
           "appearance": "texture" if tex_path else "vertex_color",
           "scale_baked": True}    # Step 3 에서 스케일을 맞추지 않는다 (asset_pool.is_scale_baked)
    if a.get("detection"):
        row["detection"] = a["detection"]
    if reg is not None:
        # 에셋 원점 = 메시 AABB 중심 (build_usd 가 AABB 중심으로 옮긴다). 정합 결과를 그 원점의 pose 로 바꾼다.
        R = Rotation.from_euler("z", reg["yaw"]).as_matrix()
        c_local = (V.max(0) + V.min(0)) / 2                  # 바닥 중심 프레임에서 AABB 중심
        T_world = pose_mat(R, R @ c_local + np.array(reg["t"]))
        T_wc = pose_mat(Rotation.from_quat(info["camera"]["orientation"]).as_matrix(), info["camera"]["position"])
        T_cam = np.linalg.inv(T_wc) @ T_world
        row["registration"] = {**reg, "pose_world": T_world.tolist(), "pose_cam": T_cam.tolist(),
                               "pose_frame": "pose_cam = 입력 카메라(opengl) 기준 에셋 원점(AABB 중심) pose"}
    print(f"[mesh] {a['category']}/{a['model']:12s} {mode:8s} 관측 h={h_w:.3f} d={d_w:.3f}  "
          f"-> extent {ext.round(3)} (h={ext[2]:.3f}, d={xy_diameter(V):.3f})"
          + (f"  yaw={np.degrees(reg['yaw']):.1f} cost={reg['cost']:.2f}" if reg else "")
          + f"  {'texture ' + str(TEX_SIZE) if tex_path else 'vertex color'}")
    return row


# ----------------------------------------------------------------------- align
def _mesh_stage_scale(spec, a, V_npz):
    """mesh 단계가 SAM3D 원본에 곱한 축별 배율 (원본 GLB 크기 대비). 정렬의 비율 사전에 쓴다."""
    import trimesh
    sc = trimesh.load(os.path.join(spec["sam3d_dir"], f"mesh_{spec['image_stem']}_{a['instance']}.glb"))
    m = trimesh.util.concatenate(list(sc.geometry.values())) if hasattr(sc, "geometry") else sc
    rot = Rotation.from_euler("x", 90, degrees=True).as_matrix().T
    return np.ptp(V_npz, axis=0) / np.ptp(np.asarray(m.vertices) @ rot, axis=0)


def stage_align(spec, assets):
    """[align] 에셋 크기를 depth 에 맞춰 만든다 (x/y/z 축별 배율을 메시에 굽는다) + 배치용 yaw/위치를 남긴다.

    에셋 생성 단계(Step 1.5)의 일부다. 여기서 정한 크기로 usd, render 를 만들고, Step 3 은 크기를 바꾸지 않고
    registration.pose_cam 의 위치와 yaw 로 놓기만 한다. 물체 종류 규칙 없이 같은 방법을 쓴다
    (our_method/utils/sam3d_pose_alignment.py):
      1) yaw 전수 탐색: 구운 텍스처 메시를 입력 카메라 그대로 렌더해 실루엣 IoU + 깊이 + 색 비교
      2) 크기: 상위 yaw 후보에서 메시 자기 축(가로 x, 세로 y, 높이 z)별 배율 + 위치를 depth 점군 3D 정합으로
         맞춘다 (관측 점 <-> 카메라에서 보이는 메시 표면, 양방향 최근접 거리). 한 시점에서 거의 안 보이는 축은
         SAM3D 원래 비율을 따르게 하는 약한 사전을 둔다.
    찾은 배율은 npz 메시(시각/충돌)에 굽는다. 그래서 이 단계 뒤에 usd, render 를 다시 돌려야 한다.
    다시 돌려도 같은 결과가 되도록 이전 배율(npz 의 align_scale)은 먼저 되돌린다.
    """
    from our_method.utils.sam3d_pose_alignment import ALIGN_METHOD, align_pose_scale, asset_pose, render_overlay
    report_path = os.path.join(spec["pool"], "sam3d_assets.json")
    report = json.load(open(report_path))
    rows = {r["model"]: r for r in report}
    depth = np.load(os.path.join(spec["inputs"], "camera_depth.npy"))[..., 0]
    info = json.load(open(os.path.join(spec["inputs"], "camera_info.json")))
    rgb_path = os.path.join(spec["inputs"], "camera_rgb.png")
    if not os.path.exists(rgb_path):        # 입력 폴더에 RGB 가 없으면 SAM3D 입력(같은 카메라)을 쓴다
        rgb_path = os.path.join(spec["sam3d_dir"], f"{spec['image_stem']}.png")
    rgb = cv2.resize(cv2.imread(rgb_path), depth.shape[::-1], interpolation=cv2.INTER_AREA)[:, :, ::-1]
    panels = []
    for a in assets:
        prompt, idx = a["instance"].rsplit("_", 1)
        mask = np.load(os.path.join(spec["sam3d_dir"], "sam3", spec["image_stem"], f"{prompt}.npy"))[int(idx)]
        mask = cv2.resize(mask.astype(np.uint8), depth.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
        npz_path = os.path.join(spec["pool"], "_build", f"{a['model']}.npz")
        d = dict(np.load(npz_path))
        prev = np.asarray(d.pop("align_scale", np.ones(3)), dtype=np.float64)
        V, CV = d["verts"] / prev, d["col_verts"] / prev               # mesh 단계 크기로 되돌린다
        native_log = -np.log(_mesh_stage_scale(spec, a, V))            # SAM3D 원래 비율로 되돌리는 log 배율
        tex = str(d["tex"]) if "tex" in d else ""
        tex = cv2.imread(tex)[:, :, ::-1] if tex and os.path.exists(tex) else None
        print(f"[align] {a['category']}/{a['model']}")
        reg = align_pose_scale(V, d["faces"], d["uvs"] if len(d["uvs"]) else None, tex, depth, info, rgb, mask,
                               native_log=native_log)
        rd = reg.pop("_renderer")
        k = np.array(reg["scale"])
        cells = render_overlay(rd, reg["yaw"], np.array(reg["t"]), scale=k)
        before = render_overlay(rd, reg["yaw_only"]["yaw"], np.array(reg["yaw_only"]["t"]))
        # 배율을 굽는다 (바닥 z=0, xy 중심 기준이라 원점 규약이 그대로 유지된다)
        d["verts"], d["col_verts"], d["align_scale"] = V * k, CV * k, k
        np.savez(npz_path, **d)
        # 배치 pose: 정합 위치와 yaw, 바닥은 받침면 (Step 3 물리 안정화도 받침면에 내려놓는다).
        # 정합이 찾은 바닥 높이(bottom_offset)는 기록만 한다 (받침면 높이 보정은 아직 안 한다)
        t_place = [reg["t"][0], reg["t"][1], reg["support_z"]]
        T_world, T_cam = asset_pose(d["verts"], reg["yaw"], t_place, info)
        row = rows[a["model"]]
        # Step 1 검출 이름. Step 3 은 이 이름으로 자기 에셋과 pose 를 찾는다 (모델 이름 기준이면 같은 카테고리
        # 물체 둘이 한 모델로 매칭될 때 섞인다)
        row["detection"] = a.get("detection") or prompt
        row["extent_mesh_stage"] = np.ptp(V, axis=0).tolist()
        row["extent"] = np.ptp(d["verts"], axis=0).tolist()
        row["size_from"] = "depth_registration"
        row["registration"] = {
            "method": ALIGN_METHOD, "yaw": reg["yaw"], "t": reg["t"], "t_place": t_place, "scale": reg["scale"],
            "cost": reg["cost"], "pc": reg["pc"], "terms": reg["terms"], "support_z": reg["support_z"],
            "bottom_offset": reg["bottom_offset"], "native_log": native_log.tolist(),
            "symmetric": reg["symmetric"], "ambiguous": reg["ambiguous"], "yaw_only": reg["yaw_only"],
            "candidates": reg["candidates"], "scale_candidates": reg["scale_candidates"], "curve": reg["curve"],
            "pose_world": T_world.tolist(), "pose_cam": T_cam.tolist(),
            "pose_frame": "pose_cam = 입력 카메라(opengl) 기준 에셋 원점(AABB 중심) pose, 바닥은 받침면. "
                          "배율은 메시에 구워져 있다"}
        cells = [cv2.resize(c, (240, 240)) for c in [cells[0], before[2], cells[1], cells[2]]]
        info_cell = np.zeros((240, 320, 3), np.uint8)
        flag = "symmetric" if reg["symmetric"] else ("AMBIGUOUS" if reg["ambiguous"] else "")
        e0, e1 = row["extent_mesh_stage"], row["extent"]
        for j, txt in enumerate([f"{a['category']} / {a['model']}", f"detection: {row['detection']}",
                                 f"yaw {np.degrees(reg['yaw']):+.1f} deg {flag}",
                                 f"scale x{k[0]:.3f} y{k[1]:.3f} z{k[2]:.3f}",
                                 f"size {e0[0]:.3f}x{e0[1]:.3f}x{e0[2]:.3f}",
                                 f"  -> {e1[0]:.3f}x{e1[1]:.3f}x{e1[2]:.3f} m",
                                 f"bottom {reg['bottom_offset'] * 100:+.1f} cm  IoU {reg['terms']['iou']:.3f}"]):
            cv2.putText(info_cell, txt, (8, 30 + 30 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
        panels.append(np.hstack([info_cell[:, :, ::-1]] + cells))
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    if panels:
        head = np.full((40, panels[0].shape[1], 3), 30, np.uint8)
        cv2.putText(head, "align: info | input crop | overlay before (yaw only, mesh-stage size) | render after "
                    "(depth xyz size) | overlay after (green = mask)", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (220, 220, 220), 1, cv2.LINE_AA)
        out = os.path.join(spec["pool"], "alignment.png")
        cv2.imwrite(out, np.vstack([head[:, :, ::-1]] + panels)[:, :, ::-1])
        print(f"[align] {out}")


# ------------------------------------------------------------------------- usd
def stage_usd(spec, a):
    """og.launch() 뒤에 불러야 한다 (pxr 이 그때 import 가능)."""
    import shutil
    # build_og_usd 는 텍스처 참조를 <OG_DATASET>/objects/<cat>/<model>/usd/textures/ 절대경로로 쓴다
    # (OG 가 암호화 USD 를 임시 폴더에 풀어서 열기 때문). import 전에 풀의 og_dataset 을 가리킨다.
    os.environ["OG_DATASET_PATH"] = os.path.join(spec["pool"], "og_dataset")
    sys.path.insert(0, KIST_OG_EXPORT)
    import build_og_usd
    build_og_usd.OG_DATASET = os.environ["OG_DATASET_PATH"]
    from omnigibson.macros import gm
    from omnigibson.utils.asset_utils import encrypt_file

    d = np.load(os.path.join(spec["pool"], "_build", f"{a['model']}.npz"))
    colors = d["colors"] if len(d["colors"]) else None
    tex = str(d["tex"]) if "tex" in d.files and str(d["tex"]) else None
    uvs = d["uvs"] if tex else None
    # 상대경로 계산이 맞도록 설치 위치와 같은 <cat>/<model>/usd/ 레이아웃으로 만든다
    plain = os.path.join(spec["pool"], "_build", "stage", a["category"], a["model"], "usd",
                         f"{a['model']}.usd")
    build_og_usd.build_usd(plain, a["category"], a["model"],
                           [(d["verts"], d["faces"], uvs, colors, tex, None)],
                           [(d["col_verts"], d["col_faces"])], mass=float(a.get("mass", 0.3)))
    dst = os.path.join(spec["pool"], "og_dataset", "objects", a["category"], a["model"],
                       "usd", f"{a['model']}.encrypted.usd")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    encrypt_file(plain, encrypted_filename=dst)
    tex_dir = os.path.join(os.path.dirname(dst), "textures")
    if os.path.isdir(tex_dir):
        shutil.rmtree(tex_dir)
    if tex:
        shutil.copytree(os.path.join(os.path.dirname(plain), "textures"), tex_dir)
    # 평문 USD(텍스처 ./textures/ 상대경로)도 같이 둔다. use_pool_usd 가 이것을 제자리에서 열어서
    # 풀 폴더를 다른 위치/머신으로 옮겨도 텍스처가 보인다 (암호화본은 이 머신 절대경로라 옮기면 깨진다).
    from our_method.utils.usd_portable import make_portable_usd
    make_portable_usd(plain, os.path.dirname(dst), a["model"])
    print(f"[usd]  {dst}  (key {gm.KEY_PATH})")


# ---------------------------------------------------------------------- render
def look_at_quat(eye, target):
    """OG viewer_camera 용 xyzw. OG 카메라도 opengl 규약(-z 전방, +y 위)."""
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    x = np.cross(f, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return Rotation.from_matrix(np.stack([x, y, -f], axis=1)).as_quat()


def stage_render(spec, assets, n_views, margin):
    import torch as th
    import omnigibson as og
    from omnigibson.scenes import Scene
    from omnigibson.objects import DatasetObject, LightObject

    use_pool_usd(spec["pool"])
    og.sim.stop()
    og.clear()
    scene = Scene(use_floor_plane=False, floor_plane_visible=False, use_skybox=False)
    og.sim.import_scene(scene)
    # 방향광 셋 (render_asset_views.py 와 같은 배치). 구형 조명은 지오메트리가 찍힌다.
    for i, (yaw_deg, pitch_deg, inten) in enumerate([(-35.0, -40.0, 3000.0),
                                                     (145.0, -25.0, 1200.0),
                                                     (60.0, -75.0, 900.0)]):
        lt = LightObject(name=f"key_light_{i}", light_type="Distant", intensity=inten, fixed_base=True)
        scene.add_object(lt)
        # render_asset_views.py 와 같은 합성: yaw 후 자기 축 기준 pitch (intrinsic)
        q = Rotation.from_euler("ZX", [yaw_deg, pitch_deg], degrees=True).as_quat()
        lt.set_position_orientation(th.tensor([0.0, 0.0, 5.0]), th.tensor(q, dtype=th.float))
    og.sim.play()
    cam = og.sim.viewer_camera
    cam.image_width, cam.image_height = W, H
    el = math.radians(spec.get("elevation", 40.0))

    for a in assets:
        cat, model = a["category"], a["model"]
        view_dir = os.path.join(spec["pool"], "objects", cat, "model", model)
        snap_dir = os.path.join(spec["pool"], "objects", cat, "snapshot")
        os.makedirs(view_dir, exist_ok=True)
        os.makedirs(snap_dir, exist_ok=True)
        obj = DatasetObject(name=f"r_{model}", category=cat, model=model,
                            visual_only=True, fixed_base=True)
        scene.add_object(obj)
        obj.set_position_orientation(th.zeros(3), th.tensor([0, 0, 0, 1.0]))
        og.sim.step()
        ext = obj.aabb_extent.cpu().numpy()
        center = obj.aabb_center.cpu().numpy()
        dist = max(float(np.linalg.norm(ext)) * margin, 0.25)
        eye = center + dist * np.array([0.0, -math.cos(el), math.sin(el)])
        cam.set_position_orientation(th.tensor(eye, dtype=th.float),
                                     th.tensor(look_at_quat(eye, center), dtype=th.float))
        for i in range(n_views):
            yaw = i * 2.0 * math.pi / N_VIEWS - math.pi
            obj.set_position_orientation(th.zeros(3), th.tensor(
                Rotation.from_euler("z", yaw).as_quat(), dtype=th.float))
            for _ in range(30 if i == 0 else 3):
                og.sim.render()
            rgb = cam.get_obs()[0]["rgb"][:, :, :3].cpu().numpy()
            if rgb.dtype != np.uint8:
                rgb = (rgb * 255).astype(np.uint8)
            Image.fromarray(rgb).save(os.path.join(view_dir, f"{model}_{i}.png"))
            if i == min(SNAPSHOT_IDX, n_views - 1):
                Image.fromarray(rgb).save(os.path.join(snap_dir, f"{cat}_{model}.png"))
        scene.remove_object(obj)
        og.sim.step()
        print(f"[render] {cat}/{model}: {n_views} views, aabb {ext.round(3)}")


# -------------------------------------------------------------------- overview
def stage_overview(spec, assets, title=None):
    """풀 루트에 소개 이미지. 행 = 물체, 열 = 정보 | SAM3D 입력 크롭 | yaw 0/25/50/75 뷰."""
    T = 240
    report = {r["model"]: r for r in json.load(open(os.path.join(spec["pool"], "sam3d_assets.json")))}
    img = cv2.imread(os.path.join(spec["sam3d_dir"], f"{spec['image_stem']}.png"))

    def put(im, txt, y, s=0.6, c=(255, 255, 255)):
        cv2.putText(im, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, s, c, 1, cv2.LINE_AA)

    def square(im, cx, cy, h):
        im = cv2.copyMakeBorder(im, h, h, h, h, cv2.BORDER_CONSTANT)
        return cv2.resize(im[cy:cy + 2 * h, cx:cx + 2 * h], (T, T))

    rows = []
    for a in assets:
        prompt, idx = a["instance"].rsplit("_", 1)
        m = np.load(os.path.join(spec["sam3d_dir"], "sam3", spec["image_stem"], f"{prompt}.npy"))[int(idx)]
        ys, xs = np.nonzero(m)
        cx, cy, h = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2, int(max(np.ptp(xs), np.ptp(ys)) * 0.65)
        crop = img.copy()
        cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(crop, cnt, -1, (0, 255, 0), 2)
        cells = [square(crop, cx, cy, h)]
        d = os.path.join(spec["pool"], "objects", a["category"], "model", a["model"])
        for k in (0, 25, 50, 75):
            v = cv2.imread(os.path.join(d, f"{a['model']}_{k}.png"))
            vy, vx = np.nonzero(v.max(2) > 12)
            cells.append(square(v, (vx.min() + vx.max()) // 2, (vy.min() + vy.max()) // 2,
                                int(max(np.ptp(vx), np.ptp(vy)) * 0.6) + 10))
        r, e = report[a["model"]], report[a["model"]]["extent"]
        info = np.zeros((T, 420, 3), np.uint8)
        put(info, f"{a['category']} / {a['model']}", 38, 0.8)
        put(info, f"SAM3D instance: {a['instance']}", 78)
        put(info, f"size: {e[0]:.3f} x {e[1]:.3f} x {e[2]:.3f} m", 113)
        put(info, f"scale from: {r['size_from']}", 148)
        put(info, f"observed: h {r['observed_height']:.3f} / d {r['observed_diameter']:.3f} m", 183)
        put(info, f"{r.get('appearance', 'vertex_color')} / mass {a.get('mass', 0.3)} kg", 218)
        rows.append(cv2.copyMakeBorder(np.hstack([info] + cells), 3, 3, 0, 0, cv2.BORDER_CONSTANT,
                                       value=(60, 60, 60)))
    w = rows[0].shape[1]
    head = np.full((112, w, 3), 30, np.uint8)
    put(head, f"SAM3D asset pool: {os.path.basename(spec['pool'])}", 38, 0.95)
    put(head, title or spec.get("title", ""), 72, 0.65)
    put(head, "columns: info | SAM3D input crop (green = SAM3 mask) | retrieval views, yaw index 0 / 25 / 50 / 75"
              "  (z_angle = i*3.6 - 180 deg)", 100, 0.55, (180, 180, 180))
    parts = [head] + rows
    if spec.get("link_categories"):
        foot = np.full((42, w, 3), 30, np.uint8)
        put(foot, "+ linked from assets/ (not SAM3D, fit-scaled): " + ", ".join(spec["link_categories"]),
            28, 0.6, (180, 180, 180))
        parts.append(foot)
    out = os.path.join(spec["pool"], "overview.png")
    cv2.imwrite(out, np.vstack(parts))
    print(f"[overview] {out}")


# ------------------------------------------------------------------------ link
def stage_link(spec):
    sys.path.insert(0, TEST_DIR)
    from asset_pool_tool import cmd_subset
    cats = spec.get("link_categories") or []
    if cats:
        cmd_subset(argparse.Namespace(include=cats, include_file=None, source=None,
                                      out=spec["pool"], link=True))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--stages", default="sam3d,mesh,align,usd,render,link,overview")
    ap.add_argument("--force-sam3d", action="store_true", help="GLB 가 있어도 SAM3D 를 다시 돌린다")
    ap.add_argument("--sam3d-list", action="store_true", help="SAM3 검출 인스턴스 목록만 보고 끝낸다")
    ap.add_argument("--only", nargs="*", help="일부 model 만")
    ap.add_argument("--n-views", type=int, default=N_VIEWS, help="확인용으로 줄일 때만. 풀은 100")
    ap.add_argument("--margin", type=float, default=1.9, help="카메라 거리 = 물체 대각선 x 이 값")
    ap.add_argument("--no-texture", action="store_true", help="텍스처를 굽지 않고 감축 메시 정점색을 쓴다")
    args = ap.parse_args()

    spec = yaml.safe_load(open(_abs(args.spec)))
    for k in ("pool", "inputs", "sam3d_dir"):
        spec[k] = _abs(spec[k])
    assets = [a for a in spec["assets"] if not args.only or a["model"] in args.only]
    for a in assets:
        assert "_" not in a["model"], f"모델명에 '_' 금지: {a['model']}"
    stages = args.stages.split(",")

    if "sam3d" in stages or args.sam3d_list:
        stage_sam3d(spec, assets, force=args.force_sam3d, list_only=args.sam3d_list)
        if args.sam3d_list:
            return
    if "mesh" in stages:
        report = [stage_mesh(spec, a, texture=not args.no_texture) for a in assets]
        os.makedirs(spec["pool"], exist_ok=True)
        with open(os.path.join(spec["pool"], "sam3d_assets.json"), "w") as f:
            json.dump(report, f, indent=2)
    if "align" in stages:
        stage_align(spec, assets)
    if "usd" in stages or "render" in stages:
        import omnigibson as og
        og.launch()
        if "usd" in stages:
            for a in assets:
                stage_usd(spec, a)
        if "render" in stages:
            stage_render(spec, assets, args.n_views, args.margin)
    if "link" in stages:
        stage_link(spec)
    if "overview" in stages:
        stage_overview(spec, assets)
    if "usd" in stages or "render" in stages:
        og.shutdown()


if __name__ == "__main__":
    main()
