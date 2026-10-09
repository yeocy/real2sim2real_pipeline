"""
asset_pool.py — Retrieval 이 후보를 끌어오는 asset 풀을 config 로 바꿀 수 있게 한다.

기존 코드는 후보 경로를 `our_method.ASSET_DIR` / `digital_cousins.ASSET_DIR` 에
직접 문자열로 박아 썼다. 그래서 풀을 바꾸려면 소스를 고쳐야 했다. 이 모듈은 그
경로 생성과 카테고리 열거를 한 곳으로 모아서, config 의 `asset_pool:` 섹션만으로
(1) 완전히 다른 asset 디렉터리를 쓰거나 (2) 기존 디렉터리에서 일부 카테고리만
쓰도록 제한할 수 있게 한다.

풀 디렉터리는 기존 assets/ 와 동일한 레이아웃을 따라야 한다:

    <root>/objects/<category>/snapshot/<category>_<model>.png   모델 선택용
    <root>/objects/<category>/model/<model>/<model>_<0..99>.png 포즈 선택용

`asset_pool` 을 지정하지 않으면(None) 기존과 100% 동일하게 동작한다.

config 예시:

    asset_pool:
      root: null                          # null -> 기존 assets/ 사용
      include_categories: [bowl, pot]     # null -> 전체 카테고리
      include_categories_file: null       # 위 목록을 파일(.json/.txt)로 줄 때
      exclude_categories: []
      strict: true                        # 화이트리스트에 실존하지 않는 카테고리가
                                          # 있으면 에러 (오타 조기 발견)
"""
import json
import os

import digital_cousins
from loguru import logger as log

from our_method.utils.dataset_utils import (
    ARTICULATION_INFO as DEFAULT_ARTICULATION_INFO,
    get_all_dataset_categories,
)


def _normalize(name):
    """카테고리 이름을 내부 표준형(underscore)으로 맞춘다. 'small bowl' -> 'small_bowl'."""
    return name.strip().replace(" ", "_")


def _load_category_list(fpath):
    """화이트리스트 파일을 읽는다. .json 은 list 또는 {"categories": [...]}, 그 외는 한 줄에 하나."""
    with open(fpath, "r") as f:
        if fpath.endswith(".json"):
            data = json.load(f)
            if isinstance(data, dict):
                data = data.get("categories", data.get("include_categories"))
            if not isinstance(data, list):
                raise ValueError(
                    f"{fpath}: JSON 은 list 이거나 'categories' 키를 가진 dict 여야 한다"
                )
            # assets/category_list.py 는 "objects/bowl" 같은 경로를 저장하므로 basename 만 취한다
            return [os.path.basename(str(c).rstrip("/")) for c in data]
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


class AssetPool:
    """Retrieval 후보의 출처. 경로 생성과 카테고리 열거를 전담한다."""

    def __init__(
        self,
        root=None,
        include_categories=None,
        include_categories_file=None,
        exclude_categories=None,
        strict=True,
        name=None,
    ):
        self.root = os.path.abspath(root) if root else digital_cousins.ASSET_DIR
        self.is_default_root = root is None
        self.name = name or ("default" if self.is_default_root else os.path.basename(self.root))
        self.strict = strict

        if not os.path.isdir(self.objects_dir):
            raise FileNotFoundError(
                f"asset pool '{self.name}': {self.objects_dir} 가 없다. "
                f"풀 루트는 <root>/objects/<category>/ 레이아웃이어야 한다."
            )

        include = list(include_categories) if include_categories else []
        if include_categories_file:
            include += _load_category_list(include_categories_file)
        self.include = {_normalize(c) for c in include} or None
        self.exclude = {_normalize(c) for c in (exclude_categories or [])}

        self._articulation_info = None
        self._available = set(os.listdir(self.objects_dir))
        if self.include and self.strict:
            missing = sorted(self.include - self._available)
            if missing:
                raise ValueError(
                    f"asset pool '{self.name}': include_categories 중 {self.objects_dir} 에 "
                    f"실제로 없는 카테고리가 있다: {missing}. "
                    f"오타가 아니라면 strict: false 로 무시할 수 있다."
                )

    # ------------------------------------------------------ articulation meta
    @property
    def articulation_info(self):
        """카테고리 -> 모델 -> (문 수, 서랍 수).

        풀 루트에 articulation_info.json 이 있으면 그것을 쓴다. 없으면 기본 assets/ 것.
        커스텀 풀이 자기 articulated 객체를 등록할 수 있어야 하므로 풀 단위로 읽는다.
        """
        if self._articulation_info is None:
            fpath = os.path.join(self.root, "articulation_info.json")
            if os.path.exists(fpath) and not self.is_default_root:
                with open(fpath, "r") as f:
                    self._articulation_info = json.load(f)
            else:
                self._articulation_info = DEFAULT_ARTICULATION_INFO
        return self._articulation_info

    # ------------------------------------------------------------------ paths
    @property
    def objects_dir(self):
        return os.path.join(self.root, "objects")

    def snapshot_dir(self, og_category):
        """모델 선택용 스냅샷 디렉터리."""
        return os.path.join(self.objects_dir, _normalize(og_category), "snapshot")

    def model_view_dir(self, og_category, og_model):
        """포즈 선택용 뷰 렌더 디렉터리."""
        return os.path.join(self.objects_dir, _normalize(og_category), "model", og_model)

    def model_view_path(self, og_category, og_model, rot_idx):
        return os.path.join(self.model_view_dir(og_category, og_model), f"{og_model}_{rot_idx}.png")

    def category_from_snapshot_path(self, fpath):
        """스냅샷 절대경로에서 카테고리를 되뽑는다. 루트가 어디든 동작한다."""
        rel = os.path.relpath(os.path.abspath(fpath), self.objects_dir)
        return rel.split(os.sep)[0]

    # ------------------------------------------------------------- categories
    def categories(self, do_not_include_categories=None, replace_underscores=True):
        """이 풀에서 retrieval 후보가 될 수 있는 카테고리 집합.

        기존 get_all_dataset_categories 와 같은 의미이되, 풀의 root 를 쓰고
        include/exclude 화이트·블랙리스트를 추가로 적용한다.
        """
        cats = get_all_dataset_categories(
            dataset_path=self.root,
            do_not_include_categories=do_not_include_categories,
            replace_underscores=False,
        )
        if self.include is not None:
            cats &= self.include
        cats -= self.exclude
        if replace_underscores:
            cats = {c.replace("_", " ") for c in cats}
        return cats

    def articulated_categories(self, do_not_include_categories=None, replace_underscores=True):
        """이 풀 안에서 articulated 인 카테고리 집합.

        articulation 메타데이터(ARTICULATION_INFO)는 기본 assets/ 에만 있으므로,
        커스텀 풀에서는 메타데이터에 없는 카테고리가 non-articulated 로 취급된다.
        """
        pool_cats = self.categories(
            do_not_include_categories=do_not_include_categories,
            replace_underscores=False,
        )
        out = set()
        for cat, models_info in self.articulation_info.items():
            if cat not in pool_cats:
                continue
            if any(sum(info) > 0 for info in models_info.values()):
                out.add(cat)
        if replace_underscores:
            out = {c.replace("_", " ") for c in out}
        return out

    # ---------------------------------------------------------------- utility
    def describe(self):
        n = len(self.categories(replace_underscores=False))
        bits = [f"root={self.root}", f"categories={n}"]
        if self.include is not None:
            bits.append(f"whitelist={len(self.include)}")
        if self.exclude:
            bits.append(f"blacklist={len(self.exclude)}")
        return f"AssetPool('{self.name}': " + ", ".join(bits) + ")"

    def validate(self, verbose=True):
        """풀 레이아웃을 점검한다. (정상 카테고리 수, 문제 목록) 을 돌려준다."""
        problems = []
        ok = 0
        for cat in sorted(self.categories(replace_underscores=False)):
            snap = self.snapshot_dir(cat)
            model_root = os.path.join(self.objects_dir, cat, "model")
            if not os.path.isdir(snap) or not os.listdir(snap):
                problems.append(f"{cat}: snapshot/ 가 없거나 비어 있다")
                continue
            if not os.path.isdir(model_root) or not os.listdir(model_root):
                problems.append(f"{cat}: model/ 가 없거나 비어 있다")
                continue
            ok += 1
        if verbose:
            log.info(f"{self.describe()} -> 정상 {ok}개, 문제 {len(problems)}개")
            for p in problems[:20]:
                log.warning(f"  {p}")
        return ok, problems


def resolve_asset_pool(spec, verbose=False):
    """config 값 -> AssetPool.

    Args:
        spec (None or str or dict or AssetPool):
            None  -> 기존 assets/ 전체 (기존 동작과 동일)
            str   -> 해당 경로를 풀 루트로 사용
            dict  -> AssetPool 생성자 인자
        verbose (bool): 해석된 풀을 로그로 남길지

    Returns:
        AssetPool
    """
    if isinstance(spec, AssetPool):
        pool = spec
    elif spec is None:
        pool = AssetPool()
    elif isinstance(spec, str):
        pool = AssetPool(root=spec)
    elif isinstance(spec, dict):
        pool = AssetPool(**{k: v for k, v in spec.items() if v is not None or k == "root"})
    else:
        raise TypeError(f"asset_pool 은 None/str/dict 여야 한다. got: {type(spec)}")

    if verbose:
        log.info(f"Retrieval asset pool: {pool.describe()}")
    return pool


# ---------------------------------------------------------------------------
# 시뮬레이터 USD 도 풀에서 불러오기
# ---------------------------------------------------------------------------
# 풀 안에 OG 데이터셋과 같은 레이아웃으로 USD 를 둔다:
#     <pool_root>/og_dataset/objects/<category>/<model>/usd/<model>.encrypted.usd (+ textures/)
# use_pool_usd() 를 부르면 DatasetObject 가 이 경로를 먼저 보고, 풀에 없는 모델만 기존
# gm.DATASET_PATH 로 간다. gm.DATASET_PATH 자체는 그대로 두므로 metadata 등은 영향이 없다.
POOL_USD_SUBDIR = "og_dataset"
DEFAULT_POOL_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "our_method_test", "asset_pools_local", "kist_twin")


def use_pool_usd(pool_root=DEFAULT_POOL_ROOT, verbose=True):
    """DatasetObject.get_usd_path 가 <pool_root>/og_dataset 를 먼저 찾게 한다. 다시 부르면 풀만 바꾼다."""
    from omnigibson.objects.dataset_object import DatasetObject

    usd_root = os.path.join(pool_root, POOL_USD_SUBDIR, "objects")
    if not os.path.isdir(usd_root):
        raise FileNotFoundError(f"풀에 USD 폴더가 없다: {usd_root}")
    current = DatasetObject.__dict__["get_usd_path"].__func__
    orig = getattr(current, "_orig", current)

    def get_usd_path(cls, category, model):
        path = os.path.join(usd_root, category, model, "usd", f"{model}.usd")
        if os.path.exists(path.replace(".usd", ".encrypted.usd")) or os.path.exists(path):
            return path
        return orig(cls, category, model)

    get_usd_path._orig = orig
    get_usd_path._pool_root = pool_root
    DatasetObject.get_usd_path = classmethod(get_usd_path)
    _load_pool_usd_unencrypted(usd_root)
    register_pool_categories(usd_root)
    baked = register_scale_baked(pool_root)
    posed = register_sam3d_poses(pool_root)
    if verbose:
        n = sum(len(os.listdir(os.path.join(usd_root, c))) for c in os.listdir(usd_root))
        log.info(f"USD 를 풀에서 먼저 불러온다: {usd_root} (모델 {n}개, 없으면 OG 데이터셋)")
        if baked:
            log.info(f"실치수가 구워진 모델 {len(baked)}개는 Step 3 에서 스케일을 맞추지 않는다: "
                     f"{sorted(f'{c}/{m}' for c, m in baked)}")
        if posed:
            log.info(f"입력 이미지에 yaw 를 맞춘 pose 가 있는 검출 {len(posed)}개는 Step 3 에서 그 pose 로 놓는다: "
                     f"{sorted(posed)}")


_POOL_USD_ROOTS = set()


def _load_pool_usd_unencrypted(usd_root):
    """풀 모델은 평문 <model>.usd 가 있으면 그것을 바로 연다 (복호화 임시 파일을 거치지 않는다).

    DatasetObject 는 항상 encrypted=True 로 .encrypted.usd 를 og.tempdir 의 임시 파일로 풀어서 연다
    (usd_object.py prebuild/_load). 그러면 USD 안의 상대경로가 임시 폴더 기준이 되어 텍스처를 못 찾는다.
    평문 USD 를 제자리에서 열면 텍스처를 ./textures/... 상대경로로 둘 수 있어 풀 폴더를 옮겨도 된다.
    """
    from omnigibson.objects.usd_object import USDObject

    _POOL_USD_ROOTS.add(os.path.abspath(usd_root))
    for meth in ("prebuild", "_load"):
        current = USDObject.__dict__[meth]
        orig = getattr(current, "_orig", current)

        def make(orig):
            def wrapped(self, *args, **kwargs):
                path = os.path.abspath(self._usd_path or "")
                plain = (self._encrypted and os.path.exists(path)
                         and any(path.startswith(r + os.sep) for r in _POOL_USD_ROOTS))
                if not plain:
                    return orig(self, *args, **kwargs)
                self._encrypted = False
                try:
                    return orig(self, *args, **kwargs)
                finally:
                    self._encrypted = True
            wrapped._orig = orig
            return wrapped

        setattr(USDObject, meth, make(orig))


_POOL_CATEGORIES = set()


def register_pool_categories(usd_root):
    """풀의 카테고리를 OG semantic class 목록에 더한다.

    OG 는 semantic class 를 gm.DATASET_PATH/objects 의 폴더 이름으로만 만든다
    (constants.semantic_class_name_to_id). 풀에서 새 이름(예: Step 1 캡션에서 만든 small_plate)을
    쓰면 렌더 시 'Class ... does not exist in the semantic class name to id mapping' 으로 죽는다.
    """
    import omnigibson.utils.constants as C

    _POOL_CATEGORIES.update(c for c in os.listdir(usd_root) if not c.startswith("."))
    current = C.get_all_object_categories
    orig = getattr(current, "_orig", current)

    def get_all_object_categories():
        return sorted(set(orig()) | _POOL_CATEGORIES)

    get_all_object_categories._orig = orig
    C.get_all_object_categories = get_all_object_categories
    C.semantic_class_name_to_id.cache_clear()
    C.semantic_class_id_to_name.cache_clear()


# SAM3D 로 만든 에셋(sam3d_asset_builder.py)은 깊이로 잰 실치수를 메시에 구워 넣는다.
# 이런 모델은 Step 3 에서 점군 bbox 에 다시 맞추면(가려진 점군이라) 비율이 틀어지므로 스케일을 바꾸지 않는다.
# 빌더가 풀 루트에 남기는 sam3d_assets.json 의 (category, model) 이 그 목록이다.
SCALE_BAKED_FILE = "sam3d_assets.json"
_SCALE_BAKED = set()


def register_scale_baked(pool_root):
    """<pool_root>/sam3d_assets.json 의 모델을 '실치수 구움' 으로 등록하고, 등록한 (category, model) 을 돌려준다."""
    fpath = os.path.join(pool_root, SCALE_BAKED_FILE)
    if not os.path.exists(fpath):
        return set()
    with open(fpath) as f:
        rows = json.load(f)
    found = {(r["category"], r["model"]) for r in rows if r.get("scale_baked", True)}
    _SCALE_BAKED.update(found)
    return found


def is_scale_baked(category, model):
    return (category, model) in _SCALE_BAKED


# sam3d_asset_builder.py 의 [align] 단계는 입력 카메라 render-and-compare 로 에셋의 yaw(z_angle)와 바닥 xy 를
# 맞춰서 행마다 registration.pose_cam (입력 카메라 opengl 기준 에셋 원점 pose)을 남긴다.
# 검출(Step 1 이름) 기준으로 등록한다. 모델 기준이면 같은 카테고리 물체 둘이 Step 2 에서 한 모델로
# 매칭될 때(04: 접시 둘 다 smallplate1) 섞인다. 검출마다 자기 메시를 만들었으므로 대응이 정해져 있다.
_SAM3D_POSES = {}


def register_sam3d_poses(pool_root):
    """<pool_root>/sam3d_assets.json 의 정렬 pose 를 검출 이름으로 등록하고, 등록한 검출 이름 집합을 돌려준다."""
    fpath = os.path.join(pool_root, SCALE_BAKED_FILE)
    if not os.path.exists(fpath):
        return set()
    with open(fpath) as f:
        rows = json.load(f)
    found = set()
    for r in rows:
        reg = r.get("registration") or {}
        if r.get("detection") and reg.get("pose_cam") is not None:
            _SAM3D_POSES[r["detection"]] = {
                "category": r["category"], "model": r["model"], "pose_cam": reg["pose_cam"],
                "yaw": reg.get("yaw"), "symmetric": reg.get("symmetric", False),
                "ambiguous": reg.get("ambiguous", False)}
            found.add(r["detection"])
    return found


def sam3d_pose(detection):
    """검출 이름 -> {category, model, pose_cam, ...} 또는 None."""
    return _SAM3D_POSES.get(detection)
