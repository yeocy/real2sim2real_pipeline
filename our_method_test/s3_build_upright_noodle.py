"""
s3_build_upright_noodle.py — 세워 놓은 라면사리 에셋(instant_noodle_block/vnoodles)을 만든다.

풀(kist_mujoco_final_el40)의 knoodles 는 knoodle 을 눕힌 채 실치수(0.099 x 0.099 x 0.0315)로
구운 것이라, Step 7 에서 그릇에 떨어뜨리면 컵 위에 눕는다. 디지털 트윈 씬(load_scene.py 의
ROW, upright=True)은 사리를 세워 담으므로, 세운 원본 vnoodle(0.034 x 0.11 x 0.11)을 같은
방식으로 구워 vnoodles 를 만든다.

knoodles 를 만든 방식(knoodle 과 속성 비교로 확인): 메시 points / extent 와 루트의
ig:nativeBB 에 축별 스케일을 곱해 bbox 를 목표 크기에 정확히 맞췄다. 법선은 그대로다.

하는 일
  1. og_dataset/objects/instant_noodle_block/vnoodle 을 복호화해 축별 스케일을 굽고
     암호화해서 .../vnoodles/usd/vnoodles.encrypted.usd 로 저장 (이미 있으면 --force 필요)
  2. 검은 배경, 앙각 40도, 1280x720 으로 뷰 100장(물체 yaw 3.6도 간격)과 스냅샷을 렌더
  3. 섹션 3 전용 로컬 풀을 만든다. 다른 카테고리는 원본 풀로 심볼릭 링크, instant_noodle_block
     만 vnoodles 로 채운다. 원본 풀은 건드리지 않는다.

실행 (acdc env, PYTHONPATH 에 airlab_twin):
  python s3_build_upright_noodle.py [--force] [--skip-usd] [--skip-render]
"""
import os
import argparse

os.environ["OMNIGIBSON_HEADLESS"] = "1"

# og.launch() 뒤에 PIL/torchvision 을 처음 import 하면 Isaac 번들 PIL 과 섞여 죽는다.
# 파이프라인 스크립트처럼 launch 전에 먼저 import 해 둔다.
import numpy as np
import torch as th
from PIL import Image
from scipy.spatial.transform import Rotation as R
import our_method.utils.transform_utils as T

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_POOL = ("/home/yeocy/robotics/LLMforMani/Simulation/real2sim2real_pipeline/"
            "our_method_test/asset_pools/kist_mujoco_final_el40")
DST_POOL = os.path.join(TEST_DIR, "asset_pools_local", "kist_el40_s3")
CATEGORY = "instant_noodle_block"
SRC_MODEL = "vnoodle"
DST_MODEL = "vnoodles"
# 세운 자세 기준 (두께, 폭, 높이). knoodles 의 [0.099, 0.099, 0.0315] 를 세운 것.
TARGET_BBOX = (0.0315, 0.099, 0.099)
N_VIEWS = 100
ELEVATION_DEG = 40.0
# 기존 풀 뷰(knoodles_*.png)에서 사리가 화면 폭의 1/3 쯤 차지하도록 맞춘 값
CAM_DIST_FACTOR = 1.6
LIGHT_INTENSITY = 150000.0


def bake_usd(force):
    from cryptography.fernet import Fernet
    from omnigibson.macros import gm
    from pxr import Usd, UsdGeom, Gf, Vt

    obj_dir = os.path.join(gm.DATASET_PATH, "objects", CATEGORY)
    src = os.path.join(obj_dir, SRC_MODEL, "usd", f"{SRC_MODEL}.encrypted.usd")
    dst_dir = os.path.join(obj_dir, DST_MODEL, "usd")
    dst = os.path.join(dst_dir, f"{DST_MODEL}.encrypted.usd")
    if os.path.exists(dst) and not force:
        print(f"[bake] {dst} 가 이미 있다. 덮어쓰려면 --force")
        return
    os.makedirs(dst_dir, exist_ok=True)

    fernet = Fernet(open(gm.KEY_PATH, "rb").read())
    tmp_src = os.path.join(dst_dir, f"_{SRC_MODEL}_tmp.usd")
    tmp_dst = os.path.join(dst_dir, f"_{DST_MODEL}_tmp.usd")
    with open(tmp_src, "wb") as f:
        f.write(fernet.decrypt(open(src, "rb").read()))

    stage = Usd.Stage.Open(tmp_src)
    root = stage.GetDefaultPrim()
    rng = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "guide"]) \
        .ComputeWorldBound(root).ComputeAlignedRange()
    size = np.array(rng.GetSize(), dtype=float)
    s = np.array(TARGET_BBOX) / size
    print(f"[bake] {SRC_MODEL} bbox {size.round(5)} -> {TARGET_BBOX}, scale {s.round(5)}")

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = np.array(mesh.GetPointsAttr().Get(), dtype=float) * s
        mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
        ext = mesh.GetExtentAttr().Get()
        if ext is not None:
            ext = np.array(ext, dtype=float) * s
            mesh.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*p) for p in ext]))
    nb = root.GetAttribute("ig:nativeBB")
    if nb and nb.Get() is not None:
        nb.Set(Gf.Vec3f(*(np.array(nb.Get(), dtype=float) * s)))

    stage.Export(tmp_dst)
    with open(dst, "wb") as f:
        f.write(fernet.encrypt(open(tmp_dst, "rb").read()))
    os.remove(tmp_src)
    os.remove(tmp_dst)

    print(f"[bake] saved {dst}")


def render_views():
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
    obj = DatasetObject(name="target", category=CATEGORY, model=DST_MODEL,
                        visual_only=True, fixed_base=True)
    scene.add_object(obj)
    # 물체 쪽으로 가까이 둔 구형 조명 셋. 기존 풀 뷰처럼 검은 배경에 밝게 찍히게.
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
    cam_pos = center + dist * np.array([np.cos(el), 0.0, np.sin(el)])
    z_axis = (cam_pos - center) / np.linalg.norm(cam_pos - center)
    x_axis = np.cross([0.0, 0.0, 1.0], z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    cam_quat = T.mat2quat(np.stack([x_axis, y_axis, z_axis], axis=1))
    og.sim.viewer_camera.set_position_orientation(th.tensor(cam_pos, dtype=th.float),
                                                  th.tensor(cam_quat, dtype=th.float))

    def shot(path, n=10):
        for _ in range(n):
            og.sim.render()
        rgb = og.sim.viewer_camera.get_obs()[0]["rgb"][:, :, :3].cpu().numpy()
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).astype(np.uint8)
        Image.fromarray(rgb).save(path)

    for i in range(N_VIEWS):
        yaw = 2 * np.pi * i / N_VIEWS
        q = R.from_euler("z", yaw).as_quat()
        obj.set_position_orientation(th.tensor([0, 0, 0], dtype=th.float), th.tensor(q, dtype=th.float))
        shot(os.path.join(view_dir, f"{DST_MODEL}_{i}.png"), n=30 if i == 0 else 5)
    # 스냅샷: 정면에서 살짝 돌린 뷰
    obj.set_position_orientation(th.tensor([0, 0, 0], dtype=th.float),
                                 th.tensor(R.from_euler("z", np.radians(30)).as_quat(), dtype=th.float))
    shot(os.path.join(snap_dir, f"{CATEGORY}_{DST_MODEL}.png"), n=10)
    print(f"[render] {N_VIEWS} views -> {view_dir}, aabb_extent={extent.round(4)}")


def build_local_pool():
    """원본 풀을 카테고리 단위로 링크하고 instant_noodle_block 만 실제 디렉터리로 둔다."""
    objects = os.path.join(DST_POOL, "objects")
    os.makedirs(objects, exist_ok=True)
    for cat in sorted(os.listdir(os.path.join(SRC_POOL, "objects"))):
        if cat == CATEGORY:
            continue
        link = os.path.join(objects, cat)
        if not os.path.lexists(link):
            os.symlink(os.path.join(SRC_POOL, "objects", cat), link)
    print(f"[pool] {DST_POOL}: {sorted(os.listdir(objects))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="vnoodles USD 가 있어도 다시 굽는다")
    ap.add_argument("--skip-usd", action="store_true")
    ap.add_argument("--skip-render", action="store_true")
    args = ap.parse_args()

    import omnigibson as og
    og.launch()
    if not args.skip_usd:
        bake_usd(args.force)
    build_local_pool()
    if not args.skip_render:
        render_views()
    og.shutdown()


if __name__ == "__main__":
    main()
