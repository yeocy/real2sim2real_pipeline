"""
build_soup_grains_asset.py — 라면 스프 알갱이 더미 에셋(soup_powder/kgrains)을 만든다.

디지털 트윈 씬(KIST_260928/scripts/kist_digital_twin_scene, ./run.sh)은 스프를 에셋 대신
반지름 1.2mm 구체 2000개를 6cm 컵(plate_soup_0)에 뿌려 만든다(load_scene.py 의
add_soup_powder). 알갱이는 USD prim 이라 GAIA retrieval 에 잡히지 않고, 씬마다 2000개를
물리로 안착시켜야 한다. 그래서 같은 조건으로 한 번 안착시킨 모양을 메시 하나로 구워
강체 에셋으로 만든다.

하는 일
  1. 시뮬레이션: 디지털 트윈과 같은 컵(plate/aewthq, 0.06 x 0.06 x 0.056)에 같은 설정
     (2000개, 반지름 1.2mm, 퍼짐 1.7cm, 격자 간격 2.4r, 바닥 2.01cm 위, 총 20g, 접촉 여유 0.5r)
     으로 알갱이를 쌓고 안착시킨 뒤 위치를 저장한다 (_build/soup_grains_settled.npy).
  2. 굽기: soup_powder/kpowderclean 을 틀로 삼아 visual 은 알갱이 2000개를 합친 메시
     (정점색 = SOUP_COLOR), collision 은 그 볼록 껍질로 바꿔 .../kgrains.encrypted.usd 로 저장.
  3. 렌더: 검은 배경, 앙각 40도, 뷰 100장 + 스냅샷을 섹션 3 로컬 풀
     (asset_pools_local/kist_el40_s3/objects/soup_powder) 에 넣는다.

실행 (acdc env, PYTHONPATH 에 airlab_twin):
  python build_soup_grains_asset.py [--force] [--skip-sim] [--skip-usd] [--skip-render]
"""
import os
import argparse

os.environ["OMNIGIBSON_HEADLESS"] = "1"

# og.launch() 뒤에 PIL/torchvision 을 처음 import 하면 Isaac 번들 PIL 과 섞여 죽는다.
import numpy as np
import torch as th
from PIL import Image
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
import our_method.utils.transform_utils as T

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
DST_POOL = os.path.join(TEST_DIR, "asset_pools_local", "kist_el40_s3")
BUILD_DIR = os.path.join(TEST_DIR, "asset_pools_local", "_build")
SETTLED_NPY = os.path.join(BUILD_DIR, "soup_grains_settled.npy")
CATEGORY = "soup_powder"
TEMPLATE_MODEL = "kpowderclean"
DST_MODEL = "kgrains"            # 모델명에 '_' 금지 (파일명을 '_' 로 가른다)

# --- 디지털 트윈 load_scene.py 값 그대로 ---
CUP = {"category": "plate", "model": "aewthq", "bounding_box": [0.06, 0.06, 0.056]}
SOUP_GRAINS = 2000
SOUP_GRAIN_R = 0.0012
SOUP_SPREAD = 0.017
SOUP_LAYER_GAP = 2.4
SOUP_CONTACT_FRAC = 0.5
SOUP_MASS = 0.02
REST_LIFT = 0.0201
SOUP_COLOR = (0.6129, 0.2400, 0.0135)

SETTLE_STEPS = 1500              # 안착 최대 스텝
SETTLE_CHECK = 100               # 이 간격마다 움직임을 본다
SETTLE_TOL = 2e-4                # 표본 최대 이동이 이보다 작으면 멈춘다 (m)
ICO_SUBDIV = 1                   # 알갱이 하나 = 42 정점 / 80 면

N_VIEWS = 100
ELEVATION_DEG = 40.0
CAM_DIST_FACTOR = 1.6
LIGHT_INTENSITY = 150000.0


def grain_spots(n, center_xy, z0, r, spread, gap):
    """load_scene._grain_spots 와 같다. 정사각 격자를 층마다 반 칸 어긋나게 쌓는다."""
    step = r * gap
    k = int(np.ceil(spread / step))
    grid = [(i * step, j * step)
            for i in range(-k, k + 1) for j in range(-k, k + 1)
            if np.hypot(i * step, j * step) <= spread]
    grid.sort(key=lambda p: np.hypot(*p))
    spots, layer = [], 0
    while len(spots) < n:
        d = (step / 2.0) if layer % 2 else 0.0
        for gx, gy in grid:
            if len(spots) >= n:
                break
            spots.append([center_xy[0] + gx + d, center_xy[1] + gy + d, z0 + layer * step])
        layer += 1
    return np.array(spots[:n], dtype=np.float64), layer


def simulate():
    """컵에 알갱이를 쌓아 안착시키고, 컵 중심·컵 바닥 기준 좌표로 저장한다."""
    import omnigibson as og
    from omnigibson.scenes import Scene
    from omnigibson.objects import DatasetObject
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    scene = Scene(use_floor_plane=True, floor_plane_visible=True, use_skybox=False)
    og.sim.import_scene(scene)
    cup = DatasetObject(name="cup", category=CUP["category"], model=CUP["model"],
                        bounding_box=CUP["bounding_box"], fixed_base=True)
    scene.add_object(cup)
    og.sim.play()
    cup.set_position_orientation(th.tensor([0, 0, 0.1], dtype=th.float), th.tensor([0, 0, 0, 1], dtype=th.float))
    og.sim.step()

    cup_lo, cup_hi = (np.array(v, dtype=np.float64) for v in cup.aabb)
    cxy = np.array(cup.get_position_orientation()[0], dtype=np.float64)[:2]
    z0 = cup_lo[2] + REST_LIFT + SOUP_GRAIN_R
    spots, layers = grain_spots(SOUP_GRAINS, cxy, z0, SOUP_GRAIN_R, SOUP_SPREAD, SOUP_LAYER_GAP)
    print(f"[sim] cup aabb lo={cup_lo.round(4)} hi={cup_hi.round(4)}, {layers}층으로 쌓음")

    stage = og.sim.stage
    root = "/World/soup_powder"
    stage.DefinePrim(root, "Scope")
    r = float(SOUP_GRAIN_R)
    ext = [Gf.Vec3f(-r, -r, -r), Gf.Vec3f(r, r, r)]
    prims = []
    for i, pos in enumerate(spots):
        sph = UsdGeom.Sphere.Define(stage, f"{root}/grain_{i}")
        sph.CreateRadiusAttr(r)
        sph.CreateExtentAttr(ext)
        prim = sph.GetPrim()
        UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*pos.tolist()))
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(SOUP_MASS / SOUP_GRAINS)
        pc = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        pc.CreateContactOffsetAttr(r * SOUP_CONTACT_FRAC)
        pc.CreateRestOffsetAttr(0.0)
        sph.CreateDisplayColorAttr([Gf.Vec3f(*SOUP_COLOR)])
        prims.append(prim)

    def read():
        return np.array([list(p.GetAttribute("xformOp:translate").Get()) for p in prims], dtype=np.float64)

    prev = read()
    for s in range(SETTLE_CHECK, SETTLE_STEPS + 1, SETTLE_CHECK):
        for _ in range(SETTLE_CHECK):
            og.sim.step()
        cur = read()
        move = np.linalg.norm(cur - prev, axis=1).max()
        print(f"[sim] step {s}: 최대 이동 {move*1000:.3f} mm, 꼭대기 {(cur[:, 2].max() - cup_lo[2])*100:.2f} cm")
        prev = cur
        if move < SETTLE_TOL:
            break

    # 컵 밖으로 나간 알갱이는 뺀다 (컵 안반지름 안, 컵 테두리 아래)
    rad = np.linalg.norm(prev[:, :2] - cxy, axis=1)
    inside = (rad < (cup_hi[0] - cup_lo[0]) / 2) & (prev[:, 2] > cup_lo[2]) & (prev[:, 2] < cup_hi[2] + 0.01)
    print(f"[sim] 컵 안 {inside.sum()} / {len(prev)} 개, 컵 테두리 {(cup_hi[2]-cup_lo[2])*100:.1f} cm, "
          f"알갱이 꼭대기 {(prev[inside, 2].max() + r - cup_lo[2])*100:.2f} cm")
    rel = prev[inside] - np.array([cxy[0], cxy[1], cup_lo[2]])
    os.makedirs(BUILD_DIR, exist_ok=True)
    np.save(SETTLED_NPY, rel)

    # 확인용 렌더: 컵을 비스듬히 위에서
    eye = np.array([cxy[0] + 0.09, cxy[1], cup_lo[2] + 0.12])
    tgt = np.array([cxy[0], cxy[1], cup_lo[2] + 0.03])
    _set_cam(og, eye, tgt)
    _shot(og, os.path.join(BUILD_DIR, "soup_grains_settled_in_cup.png"), n=30)
    print(f"[sim] saved {SETTLED_NPY}")


def icosphere(subdiv):
    t = (1.0 + 5 ** 0.5) / 2.0
    v = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
         [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
         [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
         [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    v = [np.array(p, dtype=np.float64) / np.linalg.norm(p) for p in v]
    for _ in range(subdiv):
        cache, nf = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = v[a] + v[b]
                v.append(m / np.linalg.norm(m))
                cache[key] = len(v) - 1
            return cache[key]
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        f = nf
    return np.array(v), np.array(f, dtype=np.int64)


def bake_usd(force):
    from cryptography.fernet import Fernet
    from omnigibson.macros import gm
    from pxr import Usd, UsdGeom, UsdPhysics, Gf, Vt, Sdf

    rel = np.load(SETTLED_NPY)
    r = SOUP_GRAIN_R
    obj_dir = os.path.join(gm.DATASET_PATH, "objects", CATEGORY)
    src = os.path.join(obj_dir, TEMPLATE_MODEL, "usd", f"{TEMPLATE_MODEL}.encrypted.usd")
    dst_dir = os.path.join(obj_dir, DST_MODEL, "usd")
    dst = os.path.join(dst_dir, f"{DST_MODEL}.encrypted.usd")
    if os.path.exists(dst) and not force:
        print(f"[bake] {dst} 가 이미 있다. 덮어쓰려면 --force")
        return
    os.makedirs(dst_dir, exist_ok=True)

    # 알갱이 중심을 bbox 중심이 원점이 되게 옮긴다 (kpowderclean 과 같은 규약, offsetBaseLink 0)
    lo = rel.min(axis=0) - r
    hi = rel.max(axis=0) + r
    centers = rel - (lo + hi) / 2.0
    size = hi - lo

    sv, sf = icosphere(ICO_SUBDIV)
    nv = len(sv)
    pts = (centers[:, None, :] + sv[None, :, :] * r).reshape(-1, 3)
    nrm = np.tile(sv, (len(centers), 1))
    idx = (sf[None, :, :] + (np.arange(len(centers)) * nv)[:, None, None]).reshape(-1)
    # 알갱이마다 밝기를 조금씩 달리해 가루 느낌을 낸다 (평균은 SOUP_COLOR)
    rng = np.random.default_rng(0)
    shade = np.clip(rng.normal(1.0, 0.08, len(centers)), 0.8, 1.2)
    col = np.clip(np.repeat(shade[:, None] * np.array(SOUP_COLOR)[None, :], nv, axis=0), 0, 1)

    hull = ConvexHull(pts)
    hull_idx = np.unique(hull.simplices)
    remap = -np.ones(len(pts), dtype=np.int64)
    remap[hull_idx] = np.arange(len(hull_idx))
    hull_pts = pts[hull_idx]
    hull_faces = remap[hull.simplices].reshape(-1)

    fernet = Fernet(open(gm.KEY_PATH, "rb").read())
    tmp_src = os.path.join(dst_dir, f"_{TEMPLATE_MODEL}_tmp.usd")
    tmp_dst = os.path.join(dst_dir, f"_{DST_MODEL}_tmp.usd")
    with open(tmp_src, "wb") as f:
        f.write(fernet.decrypt(open(src, "rb").read()))
    stage = Usd.Stage.Open(tmp_src)
    root = stage.GetDefaultPrim()

    def v3(a):
        return Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in a])

    ext = v3([-size / 2, size / 2])
    vis = UsdGeom.Mesh(stage.GetPrimAtPath(f"{root.GetPath()}/base_link/visuals_0"))
    col_prim = stage.GetPrimAtPath(f"{root.GetPath()}/base_link/collisions/{TEMPLATE_MODEL}_base_link_0")
    coll = UsdGeom.Mesh(col_prim)

    # visual: 템플릿 primvar(uv 등)는 정점 수가 달라지니 지우고 displayColor 만 정점 단위로 둔다
    api = UsdGeom.PrimvarsAPI(vis.GetPrim())
    for pv in api.GetPrimvars():
        if pv.GetPrimvarName() != "displayColor":
            api.RemovePrimvar(pv.GetPrimvarName())
    vis.GetPointsAttr().Set(v3(pts))
    vis.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * (len(idx) // 3)))
    vis.GetFaceVertexIndicesAttr().Set(Vt.IntArray(idx.tolist()))
    vis.GetNormalsAttr().Set(v3(nrm))
    vis.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    vis.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    vis.GetExtentAttr().Set(ext)
    dc = vis.GetDisplayColorPrimvar() or vis.CreateDisplayColorPrimvar()
    dc.Set(v3(col))
    dc.SetInterpolation(UsdGeom.Tokens.vertex)

    # collision: 더미의 볼록 껍질 하나
    cpv = UsdGeom.PrimvarsAPI(col_prim)
    for pv in cpv.GetPrimvars():
        cpv.RemovePrimvar(pv.GetPrimvarName())
    coll.GetPointsAttr().Set(v3(hull_pts))
    coll.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(hull.simplices)))
    coll.GetFaceVertexIndicesAttr().Set(Vt.IntArray(hull_faces.tolist()))
    if coll.GetNormalsAttr().HasAuthoredValue():
        coll.GetNormalsAttr().Clear()
    coll.GetExtentAttr().Set(ext)
    UsdPhysics.MeshCollisionAPI(col_prim).CreateApproximationAttr().Set(UsdPhysics.Tokens.convexHull)

    # 질량: 디지털 트윈과 같은 20g
    base = stage.GetPrimAtPath(f"{root.GetPath()}/base_link")
    UsdPhysics.MassAPI(base).CreateMassAttr().Set(float(SOUP_MASS))

    root.GetAttribute("ig:model").Set(DST_MODEL)
    root.GetAttribute("ig:nativeBB").Set(Gf.Vec3f(*map(float, size)))
    root.GetAttribute("ig:offsetBaseLink").Set(Gf.Vec3f(0, 0, 0))

    stage.Export(tmp_dst)
    with open(dst, "wb") as f:
        f.write(fernet.encrypt(open(tmp_dst, "rb").read()))
    os.remove(tmp_src)
    os.remove(tmp_dst)
    print(f"[bake] {len(centers)} 알갱이, visual {len(pts)} 정점 / {len(idx)//3} 면, "
          f"collision 볼록 껍질 {len(hull_pts)} 정점, bbox {size.round(4)} -> {dst}")


def _set_cam(og, eye, tgt):
    z_axis = (eye - tgt) / np.linalg.norm(eye - tgt)
    x_axis = np.cross([0.0, 0.0, 1.0], z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    q = T.mat2quat(np.stack([x_axis, y_axis, z_axis], axis=1))
    og.sim.viewer_camera.set_position_orientation(th.tensor(eye, dtype=th.float), th.tensor(q, dtype=th.float))


def _shot(og, path, n=10):
    for _ in range(n):
        og.sim.render()
    rgb = og.sim.viewer_camera.get_obs()[0]["rgb"][:, :, :3].cpu().numpy()
    if rgb.dtype != np.uint8:
        rgb = (rgb * 255).astype(np.uint8)
    Image.fromarray(rgb).save(path)


def render_views():
    """s3_build_upright_noodle.render_views 와 같은 조건(검은 배경, 앙각 40도)으로 찍는다."""
    import omnigibson as og
    from omnigibson.scenes import Scene
    from omnigibson.objects import DatasetObject, LightObject

    view_dir = os.path.join(DST_POOL, "objects", CATEGORY, "model", DST_MODEL)
    snap_dir = os.path.join(DST_POOL, "objects", CATEGORY, "snapshot")
    os.makedirs(view_dir, exist_ok=True)
    os.makedirs(snap_dir, exist_ok=True)

    og.sim.stop()
    og.clear()
    scene = Scene(use_floor_plane=False, floor_plane_visible=False, use_skybox=False)
    og.sim.import_scene(scene)
    obj = DatasetObject(name="target", category=CATEGORY, model=DST_MODEL, visual_only=True, fixed_base=True)
    scene.add_object(obj)
    for i, (pos, inten) in enumerate([([0.6, -0.4, 0.6], LIGHT_INTENSITY),
                                      ([0.5, 0.5, 0.5], LIGHT_INTENSITY * 0.6),
                                      ([-0.5, 0.0, 0.7], LIGHT_INTENSITY * 0.4)]):
        light = LightObject(name=f"light_{i}", light_type="Sphere", radius=0.1, intensity=inten)
        scene.add_object(light)
        light.set_position_orientation(th.tensor(pos, dtype=th.float), th.tensor([0, 0, 0, 1], dtype=th.float))
    og.sim.play()
    obj.set_position_orientation(th.tensor([0, 0, 0], dtype=th.float), th.tensor([0, 0, 0, 1], dtype=th.float))
    og.sim.step()

    extent = obj.aabb_extent.cpu().numpy()
    center = obj.aabb_center.cpu().numpy()
    dist = float(np.linalg.norm(extent)) * CAM_DIST_FACTOR
    el = np.radians(ELEVATION_DEG)
    _set_cam(og, center + dist * np.array([np.cos(el), 0.0, np.sin(el)]), center)

    for i in range(N_VIEWS):
        q = R.from_euler("z", 2 * np.pi * i / N_VIEWS).as_quat()
        obj.set_position_orientation(th.tensor([0, 0, 0], dtype=th.float), th.tensor(q, dtype=th.float))
        _shot(og, os.path.join(view_dir, f"{DST_MODEL}_{i}.png"), n=30 if i == 0 else 5)
    obj.set_position_orientation(th.tensor([0, 0, 0], dtype=th.float),
                                 th.tensor(R.from_euler("z", np.radians(30)).as_quat(), dtype=th.float))
    _shot(og, os.path.join(snap_dir, f"{CATEGORY}_{DST_MODEL}.png"), n=10)
    print(f"[render] {N_VIEWS} views -> {view_dir}, aabb_extent={extent.round(4)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="kgrains USD 가 있어도 다시 굽는다")
    ap.add_argument("--skip-sim", action="store_true", help="저장된 안착 결과(npy)를 쓴다")
    ap.add_argument("--skip-usd", action="store_true")
    ap.add_argument("--skip-render", action="store_true")
    args = ap.parse_args()

    import omnigibson as og
    og.launch()
    if not args.skip_sim:
        simulate()
    if not args.skip_usd:
        bake_usd(args.force)
    if not args.skip_render:
        render_views()
    og.shutdown()


if __name__ == "__main__":
    main()
