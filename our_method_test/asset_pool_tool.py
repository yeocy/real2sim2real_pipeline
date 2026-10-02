#!/usr/bin/env python
"""
asset_pool_tool.py — retrieval asset 풀을 점검하고 새로 만든다.

config 의 `asset_pool:` 에 넣을 값을 실제로 돌려보기 전에 확인하는 용도.

  # config 에 적힌 풀이 유효한지, 카테고리가 몇 개나 잡히는지 확인
  python asset_pool_tool.py check --config configs/kist_ramen.yaml

  # 임의 풀 직접 확인
  python asset_pool_tool.py check --root /path/to/pool --include bowl pot

  # 기존 assets/ 에서 카테고리 일부만 골라 심볼릭 링크 풀을 만든다
  #   (스냅샷/뷰 렌더를 복사하지 않으므로 즉시 만들어지고 디스크도 안 먹는다)
  python asset_pool_tool.py subset --out /path/to/new_pool \
      --include bowl pot frying_pan stove --link

  # 카테고리 이름 검색 (화이트리스트 작성용)
  python asset_pool_tool.py search bowl pot noodle
"""
import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import digital_cousins  # noqa: E402
from our_method.utils.asset_pool import AssetPool, resolve_asset_pool  # noqa: E402

DEFAULT_OBJECTS = os.path.join(digital_cousins.ASSET_DIR, "objects")


def cmd_check(args):
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        spec = cfg.get("asset_pool", None)
        print(f"[config] {args.config} -> asset_pool = {spec}")
    else:
        spec = {
            "root": args.root,
            "include_categories": args.include or None,
            "exclude_categories": args.exclude or [],
            "strict": not args.no_strict,
        }

    pool = resolve_asset_pool(spec)
    print(pool.describe())
    ok, problems = pool.validate(verbose=False)
    print(f"레이아웃 정상 카테고리: {ok}")
    if problems:
        print(f"문제 {len(problems)}건:")
        for p in problems[:30]:
            print(f"  - {p}")
        if len(problems) > 30:
            print(f"  ... 외 {len(problems)-30}건")
    cats = sorted(pool.categories(replace_underscores=False))
    print(f"후보 카테고리 {len(cats)}개: {cats[:25]}{' ...' if len(cats) > 25 else ''}")
    art = sorted(pool.articulated_categories(replace_underscores=False))
    print(f"그중 articulated {len(art)}개: {art[:15]}{' ...' if len(art) > 15 else ''}")
    return 0 if not problems else 1


def cmd_subset(args):
    include = list(args.include or [])
    if args.include_file:
        with open(args.include_file) as f:
            if args.include_file.endswith(".json"):
                data = json.load(f)
                include += [os.path.basename(str(c).rstrip("/")) for c in data]
            else:
                include += [ln.strip() for ln in f if ln.strip()]
    include = [c.replace(" ", "_") for c in include]
    if not include:
        sys.exit("--include 또는 --include-file 로 카테고리를 지정해야 한다")

    src_objects = args.source or DEFAULT_OBJECTS
    dst_objects = os.path.join(args.out, "objects")
    os.makedirs(dst_objects, exist_ok=True)

    made, missing = [], []
    for cat in include:
        s = os.path.join(src_objects, cat)
        if not os.path.isdir(s):
            missing.append(cat)
            continue
        d = os.path.join(dst_objects, cat)
        if os.path.lexists(d):
            made.append(cat)
            continue
        if args.link:
            os.symlink(os.path.abspath(s), d)
        else:
            import shutil
            shutil.copytree(s, d)
        made.append(cat)

    # articulation 메타데이터는 기본 assets/ 에서 읽으므로 풀 루트에도 같이 놔둔다
    for meta in ["articulation_info.json",
                 "articulated_obj_valid_rotation_angle_range.json",
                 "_tmp_reorientation_info.json"]:
        s = os.path.join(digital_cousins.ASSET_DIR, meta)
        d = os.path.join(args.out, meta)
        if os.path.exists(s) and not os.path.lexists(d):
            os.symlink(s, d) if args.link else __import__("shutil").copy(s, d)

    print(f"생성: {args.out}  (카테고리 {len(made)}개, {'symlink' if args.link else 'copy'})")
    if missing:
        print(f"원본에 없어 건너뜀 {len(missing)}개: {missing}")
    print("\nconfig 에 아래를 넣으면 된다:\n")
    print("asset_pool:")
    print(f"  root: {os.path.abspath(args.out)}")
    return 0


def cmd_search(args):
    src = args.source or DEFAULT_OBJECTS
    cats = sorted(os.listdir(src))
    for term in args.terms:
        hits = [c for c in cats if term.lower().replace(" ", "_") in c.lower()]
        print(f"[{term}] {len(hits)}개: {hits}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="풀 유효성 점검")
    c.add_argument("--config", help="yaml config 에서 asset_pool 을 읽어 점검")
    c.add_argument("--root", default=None)
    c.add_argument("--include", nargs="*")
    c.add_argument("--exclude", nargs="*")
    c.add_argument("--no-strict", action="store_true")
    c.set_defaults(func=cmd_check)

    s = sub.add_parser("subset", help="기존 풀에서 카테고리 일부만 뽑아 새 풀 생성")
    s.add_argument("--out", required=True)
    s.add_argument("--source", default=None, help="원본 objects/ 경로 (기본: assets/objects)")
    s.add_argument("--include", nargs="*")
    s.add_argument("--include-file")
    s.add_argument("--link", action="store_true", help="복사 대신 심볼릭 링크 (권장)")
    s.set_defaults(func=cmd_subset)

    g = sub.add_parser("search", help="카테고리 이름 검색")
    g.add_argument("terms", nargs="+")
    g.add_argument("--source", default=None)
    g.set_defaults(func=cmd_search)

    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
