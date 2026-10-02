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
