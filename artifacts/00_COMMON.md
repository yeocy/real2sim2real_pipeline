# 공통 안내 — 남은 세션이 모두 먼저 읽을 것

최종 갱신: 2026-10-03 (섹션 1·3·5 완료, 라면 스프 에셋 추가 반영)

이 저장소(`airlab_twin`)는 KIST 디지털 트윈 씬 사진 한 장(RGB + GT depth)으로 OmniGibson 시뮬레이션 씬을 재구성하는 GAIA 파이프라인이다.

| 섹션 | 상태 | 문서 |
|---|---|---|
| 1. 인덕션 배치 | ✅ 완료 (`support_yaw_align`) | — |
| 2. 가려진 부분 depth 채우기 | ⏳ 남음 | [02_DEPTH_OCCLUSION_COMPLETION.md](02_DEPTH_OCCLUSION_COMPLETION.md) |
| 3. 그릇 안 물체 Step 4~7 배치 | ✅ 완료 | — |
| 4. SAM3D → asset pool 생성 도구 | ⏳ 남음 | [04_SAM3D_ASSET_POOL_BUILDER.md](04_SAM3D_ASSET_POOL_BUILDER.md) |
| 5. 최종 씬 물리 반영 (관통 제거·안착) | ✅ 완료 (`physics_settle`) | — |

---

## 1. 실행 환경

```bash
conda activate acdc
cd /home/yeocy/robotics/LLMforMani/Simulation/airlab_twin/our_method_test
export PYTHONPATH=/home/yeocy/robotics/LLMforMani/Simulation/airlab_twin:$PYTHONPATH   # 필수
source <(grep "^export OPENAI_API_KEY" ../chatgpt_api_key)                             # 키를 출력하지 말 것
export GAIA_NO_FIT_SCALE=1                                                              # 풀이 실치수라 점군에 맞춰 늘리지 않음
python our_models_task_generation.py --config configs/<내_config>.yaml
```

- **PYTHONPATH를 빼면 원본 폴더(`real2sim2real_pipeline`)의 코드가 import된다.** `acdc` env에 원본 경로로 editable 설치돼 있기 때문이다.
- `assets/`, `checkpoints/`, `deps/`, `chatgpt_api_key`, `our_method_test/asset_pools/`, `our_method_test/inputs_kist_twin_full/`은 원본 폴더로 가는 심볼릭 링크다. 링크 대상을 수정하면 원본도 바뀐다. 자세한 건 [SETUP_SYMLINKS.md](SETUP_SYMLINKS.md) 참고.
  - 새 에셋 USD(`vnoodles`, `kgrains`)도 `deps/.../og_dataset/objects/` 아래에 만들었다. 즉 실제 파일은 원본 폴더에 있다.
- GPU: RTX 4070 Ti 12GB 한 장. **두 세션이 동시에 OmniGibson/모델을 돌리면 OOM이 날 수 있다.** 무거운 실행 전에 `nvidia-smi`를 확인한다.
- 실행 시간: Step 1~3이 약 10분 걸린다(Step 2가 6분).
- 그 밖의 선택 환경변수(기본은 꺼짐): `GAIA_PC_OUTLIER_FILTER=1`(물체 점군의 depth 이상치 제거, `real_scene_generation.py` `_load_obj_pc`), `GAIA_ROBUST_FLOOR=1`(바닥 평면 적합 강화, `extraction.py`)

## 2. 입력과 현재 기준 결과

- 입력: `our_method_test/inputs_kist_twin_full/` — `camera_rgb.png`, `camera_depth.npy`(GT, 미터), `camera_info.json`(GT intrinsics, 1600×1600, f≈1931)
- **현재 기준 결과(통합본, 건드리지 말 것): `our_method_test/acdc_out_s5_physics/`**, config `configs/s5_physics.yaml`
  - `step_1_output/`, `step_2_output/`: 처음 실행(`acdc_out_el40_recap`)과 내용이 같다.
  - `step_3_output/`: 섹션 1(인덕션 yaw를 앞 테이블에 정렬, `support_yaw_align`, 인덕션 +180°)과 섹션 5(관통 해소 + 중력 안착)를 적용했다.
  - `task_*` 폴더: 섹션 3(그릇 안에 면 블록, 파, 만두, 라면 스프를 배치)과 섹션 5를 적용했다.
  - 물리 검증: `task_scene_generation/physics_check.json`, `physics_settle_report.json`. 300스텝 안착 후 모든 물체의 이동이 1 mm 이하다. 비교 이미지는 `final_cam_view.png`, `final_closeup*.png`
  - config의 `pipeline_steps`는 Step 3·7만 켜져 있다. 다른 Step을 다시 돌릴 때는 해당 플래그를 켠다.
- 처음 실행 결과(섹션 1·3·5 이전): `our_method_test/acdc_out_el40_recap/`. 비교용으로만 남겨 둔다.
- Step 6(그릇 안 내용물 retrieval) 풀: `our_method_test/asset_pools_local/kist_el40_s3/`. 원본 풀을 카테고리별로 링크하고, 다음 두 가지만 실제 파일로 둔 풀이다.
  - `instant_noodle_block/vnoodles`: 세운 라면사리. 생성 스크립트 `s3_build_upright_noodle.py`
  - `soup_powder/kgrains`: 라면 스프. 디지털 트윈(`KIST_260928/.../load_scene.py`의 `add_soup_powder`)과 같은 조건으로 1.2 mm 알갱이 2000개를 6 cm 컵에 안착시켜, 그 모양을 강체 하나로 구운 에셋이다. 크기 4.3×4.3×3.6 cm, 20 g, 충돌은 볼록 껍질 하나. 생성 스크립트 `build_soup_grains_asset.py`
- 리사이즈 단계(`task_object_resizing.py`)는 섹션 3에서 제거됐다. 파이프라인은 Step 4·5 → 6 → 7로 이어진다.
- 도구:
  - `check_scene_physics.py <scene_info.json>`: 관통과 안착 상태 측정
  - `view_scene.py <scene_info.json>`: 물리를 켜고 GUI로 보기

## 3. 세션 간 충돌을 막는 규칙

1. **출력 폴더를 분리한다.** 기준 결과를 복사해서 자기 폴더에서 작업한다.
   ```bash
   cp -r acdc_out_s5_physics acdc_out_s<N>_<이름>
   cp configs/s5_physics.yaml configs/s<N>_<이름>.yaml   # paths.save_dir 을 자기 폴더로
   grep -rl acdc_out_s5_physics acdc_out_s<N>_<이름> | xargs sed -i 's#/acdc_out_s5_physics/#/acdc_out_s<N>_<이름>/#g'
   ```
   - 복사한 폴더 안 json에는 원래 폴더의 절대경로가 들어 있다. 위의 세 번째 줄로 자기 폴더를 가리키게 바꾼다.
   - 각 Step은 `save_dir/step_K_output/...`을 읽으므로 `pipeline_steps`에서 필요한 Step만 `true`로 켜면 된다.
2. **파일 소유권**: 아래 표에 없는 파일을 고쳐야 하면 고치기 전에 사용자에게 알린다. 완료된 섹션이 고친 `real_scene_generation.py`, `task_*.py`, `physics_settle.py`는 기존 동작을 깨지 않는 범위에서만 건드린다.

   | 섹션 | 주로 수정하는 파일 |
   |---|---|
   | 2 | `our_method/pipeline/extraction.py`, `our_method/utils/processing_utils.py`, 새 모듈. Step 3 연결부(`real_scene_generation.py`의 `_load_obj_pc`)는 최소 수정 |
   | 4 | 새 스크립트(`our_method_test/build_asset_pool.py` 등), `our_method/utils/asset_pool.py`, `our_method_test/asset_pool_tool.py` |

3. **새 기능은 config 옵션이나 환경변수로 켜고 끈다.** 끈 상태에서는 현재 기준 결과가 그대로 재현돼야 한다.
4. 원본 폴더(`real2sim2real_pipeline`), 기준 결과 폴더 두 개, 원본 풀(`asset_pools/kist_mujoco_final_el40`), `asset_pools_local/kist_el40_s3`의 기존 카테고리는 수정하지 않는다.
5. 커밋은 사용자가 요청할 때만 한다. 섹션 1·3·5와 스프 에셋 변경도 아직 커밋 전이다.

## 4. 알려진 무해한 로그

- Step 3·7의 `'NoneType' object has no attribute 'GetCamera'`, `"/World/viewer_camera" is not a valid Usd.Prim`: Isaac 뷰포트 콜백 경고다.
- 마지막의 `Fatal Python error: Segmentation fault`(exit 139): Isaac Sim이 플러그인을 언로드할 때 난다. 해당 Step의 `SUCCESS ...` 로그가 찍혔다면 결과는 정상이다.
- OmniGibson 스크립트에서 `print` 출력이 사라지면 `PYTHONUNBUFFERED=1`을 준다. 종료할 때 stdout 버퍼가 버려진다.
