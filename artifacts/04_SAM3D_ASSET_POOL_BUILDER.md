# 섹션 4 — SAM3D 결과물로 asset pool을 만드는 도구

> 먼저 [00_COMMON.md](00_COMMON.md)를 읽을 것. 이 섹션은 파이프라인을 돌리지 않아도 시작할 수 있다. 렌더링에 GPU가 필요하니 다른 세션과 겹치지 않게 한다.

## 목표
지금 쓰는 retrieval 풀 `kist_mujoco_final_el40`은 SAM3D로 만든 메시를 **손으로** 풀 레이아웃에 맞춰 넣은 것이다. 이 풀을 만든 스크립트는 저장소 안에서 찾지 못했다.

**출발점 (섹션 3에서 생김)**: `our_method_test/s3_build_upright_noodle.py`가 모델 **하나**(`instant_noodle_block/vnoodles`)에 대해 전 과정을 이미 구현했다. 이걸 일반화하는 것이 이 섹션의 핵심이다.
- USD 복호화 → 축별 스케일 굽기(메시 points/extent + 루트 `ig:nativeBB`) → 재암호화 (`bake_usd`)
- 검은 배경, 앙각 40°, 1280×720, yaw 3.6° 간격 100장 + 스냅샷 렌더 (`render_views`)
- 로컬 풀 생성: 바꾼 카테고리만 실제 파일로 두고, 나머지는 원본 풀로 심볼릭 링크 → `asset_pools_local/kist_el40_s3/`
- 상수(`SRC_POOL`, `N_VIEWS=100`, `ELEVATION_DEG=40`, 카테고리/모델명)가 하드코딩돼 있다.
**SAM3D 결과물(메시/USD) 폴더를 주면 GAIA가 바로 쓸 수 있는 asset pool을 자동으로 만들어 주는 도구**를 만든다.

## 풀이 갖춰야 하는 것

### 1) 렌더 이미지 (retrieval용) — `our_method/utils/asset_pool.py` 상단 docstring 참고
```
<pool_root>/objects/<category>/snapshot/<category>_<model>.png      # 모델 선택용 대표 이미지 1장
<pool_root>/objects/<category>/model/<model>/<model>_<0..99>.png    # 포즈 선택용 yaw 100장
```
- 기존 풀은 snapshot 폴더에 `<model>_<idx>.png`도 같이 들어 있다. 필요한지 코드(`dataset_utils.py:140`, `matching.py:459`)로 확인한다.
- **모델명에 `_`를 쓰면 안 된다.** 코드가 파일명을 `_`로 가른다.
- **렌더 앙각 40°.** 입력 사진이 물체를 29~48°로 내려다본다. 기존 기본 풀처럼 20°로 렌더하면 평평한 가구의 yaw 매칭이 크게 틀어진다(`kist_el40_recap.yaml`의 asset_pool 주석 참고). 앙각은 인자로 받는다.
- yaw 100장의 각도 규약(인덱스 → z_angle)은 `matching.py`가 쓰는 규약과 **반드시** 같아야 한다. 예: 인덕션 `z_angle 1.2566 = kinductionclean_70.png`. 코드에서 규약을 확인하고 맞춘다.

### 2) 시뮬레이터용 에셋 (배치용)
- OmniGibson이 `og_dataset/objects/<category>/<model>/usd/<model>.encrypted.usd`(+ `textures/`)에서 로드한다.
- 현재 SAM3D 에셋의 원본 위치: `/home/yeocy/robotics/Kist/KIST_260928/scripts/kist_digital_twin_scene/assets/og_dataset/objects/`
  (`deps/OmniGibson/omnigibson/data/og_dataset/objects/<cat>/<model>`이 이곳을 가리키는 심볼릭 링크다. `deps/`는 원본 폴더 공유이므로 링크를 추가하면 원본에도 반영된다는 점에 주의.)
- **스케일을 메시에 구워 넣는다.** 씬에서 쓰는 실치수를 USD에 반영하고 `GAIA_NO_FIT_SCALE=1`로 쓴다. 같은 메시가 여러 크기로 쓰이면 모델을 나눈다(예: `aewthqdish` 0.16 m / `aewthqnoodle` 0.126 m / `aewthqcup` 0.06 m).
- 관절(articulation) 정보: `assets/articulation_info.json` 등. SAM3D 결과는 강체라 대부분 해당 없지만, 로딩 코드가 요구하는 항목이 있는지 확인한다.

## 참고 자료
| 위치 | 내용 |
|---|---|
| `our_method_test/asset_pools/kist_mujoco_final_el40/` | **정답 예시** (카테고리 10개 / 모델 13개) |
| `our_method_test/s3_build_upright_noodle.py` | **단일 모델용 구현** (스케일 굽기 + 렌더 + 로컬 풀). 일반화 대상 |
| `our_method_test/asset_pools_local/kist_el40_s3/` | 위 스크립트의 결과. 섹션 3의 Step 6이 쓰는 중이니 **덮어쓰지 말 것** |
| `our_method_test/asset_pool_tool.py` | 기존 도구: `check`(풀 검증), `subset`(기존 assets 일부를 링크), `search` |
| `our_method/utils/asset_pool.py` | `AssetPool`, `resolve_asset_pool`, `validate()` |
| `our_method/utils/dataset_utils.py` | 카테고리/모델 열거, 파일명 파싱 |
| `/home/yeocy/robotics/Kist/KIST_260928/scripts/kist_digital_twin_scene/` | SAM3D 에셋 정리 스크립트들 (`make_clean_induction.py`, `make_clean_purifier.py`, `localize_textures.py`, `setup_assets.sh`, `_views/` 등). **읽기만** 할 것 |
| `our_method_test/export_usd_to_urdf.py` | USD → URDF 변환 예시 (템플릿 수준) |

## 만들 도구 (제안)
`our_method_test/build_asset_pool.py` 또는 `asset_pool_tool.py`의 새 하위 명령 `build`:
```bash
python asset_pool_tool.py build \
    --src <SAM3D 결과 폴더 또는 og_dataset/objects 경로> \
    --out our_method_test/asset_pools/<새 풀 이름> \
    --elevation 40 --n-views 100 \
    [--scale-spec scales.yaml]     # 모델별 실치수 / 분할 규칙
```
1. 입력 폴더의 카테고리와 모델을 열거하고 이름을 검사한다(`_` 금지).
2. (옵션) 스케일 스펙에 따라 USD에 스케일을 굽고 모델을 분할한다.
3. OmniGibson(headless)에서 모델별로 앙각 N°, yaw 100장, snapshot 1장을 렌더한다. 배경과 조명은 기존 풀과 맞춘다.
4. 끝나면 `asset_pool_tool.py check --root <out>`이 문제 0건이어야 한다.

## 완료 기준
- **재현성 검증**: 같은 원본으로 `kist_mujoco_final_el40`과 같은 구성의 풀을 새로 만든다. 그 풀로 `kist_el40_recap.yaml`의 Step 2~3을 다시 돌려서, 매칭 결과(카테고리/모델/`z_angle`)가 기준 결과와 같거나 더 낫다.
- 새 카테고리 하나(예: KIST 폴더의 `pot_lid` 또는 `seasoning_packet`)를 추가해서 풀을 만들고, `check`를 통과한다.
- 사용법을 이 문서 아래나 스크립트 docstring에 적는다.

## 주의할 점
- 기존 풀(`asset_pools/kist_mujoco_final_el40`), `asset_pools_local/kist_el40_s3`, KIST 원본 폴더는 **덮어쓰지 않는다.** 새 이름으로 만든다.
- 일반화한 뒤 `s3_build_upright_noodle.py`를 새 도구의 한 사용 예로 대체할 수 있다. 다만 기존 스크립트는 지우지 말고, 새 도구로 같은 풀을 재현할 수 있다는 것까지 확인한 뒤 사용자에게 정리할지 묻는다.
- 재현성 검증은 현재 기준 결과(`acdc_out_s5_physics`, config `s5_physics.yaml`)로 한다.
- 참고 구현이 하나 더 있다: `our_method_test/build_soup_grains_asset.py`. 알갱이 2000개를 시뮬레이션으로 안착시킨 모양을 메시 하나로 합치고, 템플릿 USD(`kpowderclean`)의 visual·collision·질량·`ig:nativeBB`를 바꿔 새 에셋을 만든다. "메시를 새로 만들어 넣는" 경우의 예시다.
- `asset_pools/`는 원본 폴더로 가는 심볼릭 링크다. 새 풀도 결국 원본 폴더에 생성된다는 점을 알고 진행한다.
