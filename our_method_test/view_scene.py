"""
scene_info.json(Step 3 scene_0_info.json 또는 Step 7 task_scene_generation/scene_info.json)을 OmniGibson GUI 로 띄운다.
충돌과 중력을 켜고 fixed_categories(테이블)만 고정한 채 창을 닫을 때까지 시뮬레이션을 계속 돌린다.

사용:
    python view_scene.py acdc_out_s5_physics/task_scene_generation/scene_info.json
"""
import os
import json
import argparse

import omnigibson as og
from our_method.utils.asset_pool import use_pool_usd, DEFAULT_POOL_ROOT  # launch 전에 import (PIL 충돌)
from omnigibson.macros import gm

from our_method.utils.physics_settle import load_scene, PHYSICS_SETTLE_DEFAULTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_info")
    ap.add_argument("--fixed_categories", nargs="*", default=list(PHYSICS_SETTLE_DEFAULTS["fixed_categories"]))
    ap.add_argument("--no_physics", action="store_true", help="물리 없이 렌더링만 (visual_only + fixed_base)")
    ap.add_argument("--usd_pool", default=None,
                    help="USD 를 먼저 불러올 풀 (기본: 통합 풀 kist_twin). 'none' 이면 OG 데이터셋만 쓴다")
    args = ap.parse_args()

    with open(args.scene_info) as f:
        scene_info = json.load(f)

    gm.HEADLESS = False
    og.launch()
    if (args.usd_pool or "").lower() != "none":
        use_pool_usd(args.usd_pool or DEFAULT_POOL_ROOT)
    load_scene(scene_info, "static" if args.no_physics else "dynamic", args.fixed_categories)
    print(f"[view_scene] {len(scene_info['objects'])} objects loaded from {os.path.abspath(args.scene_info)}. "
          f"창을 닫거나 Ctrl+C 로 종료.")
    try:
        while og.app.is_running():
            if args.no_physics:
                og.sim.render()
            else:
                og.sim.step()
    except KeyboardInterrupt:
        pass
    og.shutdown()


if __name__ == "__main__":
    main()
