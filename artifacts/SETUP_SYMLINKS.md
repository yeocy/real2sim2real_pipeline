# airlab_twin 실행 환경 구성 기록

- 작성일: 2026-10-03
- 원본 폴더: `/home/yeocy/robotics/LLMforMani/Simulation/real2sim2real_pipeline`
- 클론 폴더: `/home/yeocy/robotics/LLMforMani/Simulation/airlab_twin`
- 원격 저장소: `https://github.com/yeocy/real2sim2real_pipeline.git`

원본 폴더를 GitHub에 푸시한 뒤 `airlab_twin`으로 다시 클론했다. 용량이 크거나 `.gitignore`로 제외된 실행 필수 파일은 클론에 포함되지 않으므로, 원본 폴더를 가리키는 **심볼릭 링크**로 연결했다.

---

## 1. Git 클론으로 가져온 파일 (git 추적 파일, 총 207개)

| 경로 | 파일 수 | 내용 |
|---|---|---|
| `digital_cousins/` | 37 | ACDC(Digital Cousins) 원본 패키지 코드 |
| `our_method/` | 73 | 우리 방법 파이프라인 코드 |
| `our_examples/` | 17 | 데모 수집 / 데이터셋 처리 / 학습 / 평가 스크립트 |
| `examples/` | 8 | ACDC 원본 예제 스크립트 |
| `our_method_test/configs/` | 53 | 실험 설정 yaml |
| `our_method_test/inputs/` | 7 | 기본 입력 (RGB, depth, camera_info, scene_info) |
| `our_method_test/*.py` | 4 | `asset_pool_tool.py`, `export_usd_to_urdf.py`, `our_models_task_generation.py`, `re_axis_change.py` |
| 루트 파일 | — | `setup.py`, `install.sh`, `requirements.txt`, `model_check.py`, `README.md`, `LICENSE`, `.gitignore`, `.vscode/` |

---

## 2. 심볼릭 링크로 연결한 파일

모든 링크는 원본 폴더(`real2sim2real_pipeline/`)의 같은 경로를 가리킨다.

| 링크 경로 (airlab_twin 기준) | 크기 | 내용 | 필요한 이유 |
|---|---|---|---|
| `assets/` | 46G | OmniGibson 오브젝트·로봇 에셋, articulation 관련 json | `digital_cousins.ASSET_DIR = {REPO_DIR}/assets` |
| `checkpoints/` | 5.0G | GroundingDINO, SAM2, DepthAnything V2 가중치 | `digital_cousins.CHECKPOINT_DIR = {REPO_DIR}/checkpoints` |
| `deps/` | 65G | OmniGibson, GroundingDINO, segment-anything-2, UniDepth, PerspectiveFields, robomimic, dinov2, Depth-Anything-V2 | 외부 의존 라이브러리 |
| `chatgpt_api_key` | — | GPT API 키 | GPT 호출 단계 |
| `our_method_test/asset_pools/` | — | `kist_ramen`, `kist_ramen_scene`, `kist_mujoco_final_el40`, `lumen_real`, `lumen_watering`, `ramen` | config의 asset pool `root` |
| `our_method_test/inputs_kist_front_high/` | — | KIST 정면(높은 시점) 입력 | `kist_front_high*.yaml`, `kist_t*.yaml` |
| `our_method_test/inputs_kist_front_left/` | — | KIST 좌측 시점 입력 | 관련 config |
| `our_method_test/inputs_kist_twin_full/` | — | KIST twin 전체 입력 | 관련 config |

### Git 제외 처리
심볼릭 링크는 git이 "파일"로 인식하므로 `.gitignore`의 `assets/`처럼 슬래시로 끝나는 규칙에 걸리지 않는다. 그래서 **`.git/info/exclude`**(로컬 전용, 커밋되지 않음)에 다음을 추가했다.

```
/assets
/checkpoints
/deps
/chatgpt_api_key
/our_method_test/asset_pools
/our_method_test/inputs_kist_*
```

---

## 3. 연결하지 않은 것 (의도적으로 제외)

| 항목 | 제외 이유 |
|---|---|
| `our_method_test/acdc_output*`, `kist_front_high/`, `libero_goal/`, `libero_spatial/`, `tamp/`, `AnyTask/`, `logs/` | 실험 **출력** 폴더. 링크하면 새 실행 결과가 원본 폴더에 섞여 저장됨 |
| `logs/` (루트) | 과거 실행 로그 |
| `GAIA_Paper_Submitted/`, `AnyTask_Submitted/` | 논문 제출본, 실행과 무관 |
| `execute` | 개인 명령어 메모 (API 키 포함) |
| `*.bak`, `*.bak2`, `*.zip`, `digital_cousins.egg-info/` | 백업·압축·빌드 산출물 |

이전 단계 출력이 필요하면(예: `1_collect_demos.py`에 `acdc_output/step_3_output` 사용) 해당 폴더만 추가로 링크하면 된다.

```bash
ln -s /home/yeocy/robotics/LLMforMani/Simulation/real2sim2real_pipeline/our_method_test/acdc_output \
      /home/yeocy/robotics/LLMforMani/Simulation/airlab_twin/our_method_test/acdc_output
```

---

## 4. 실행 전 필수: PYTHONPATH 설정

`acdc` conda 환경에는 `digital_cousins`, `omnigibson`, `sam2`, `groundingdino`, `unidepth`, `perspective2d`, `robomimic`가 **원본 폴더 경로로 editable 설치**되어 있다. 아무 설정 없이 실행하면 `airlab_twin`이 아니라 **원본 폴더의 코드**가 import된다.

```bash
conda activate acdc
export PYTHONPATH=/home/yeocy/robotics/LLMforMani/Simulation/airlab_twin:$PYTHONPATH
```

확인 결과:

| 설정 | `digital_cousins` / `our_method` import 위치 |
|---|---|
| PYTHONPATH 없음 | `real2sim2real_pipeline/...` (원본) |
| PYTHONPATH 설정 | `airlab_twin/...` (클론) ✅ |

`deps/` 아래 라이브러리는 계속 원본 경로에서 로드되지만, 어차피 심볼릭 링크로 같은 파일을 공유하므로 문제없다.

---

## 5. 주의사항

- **원본 폴더를 삭제·이동하면 안 된다.** 심볼릭 링크와 아래 절대경로가 모두 깨진다.
- `our_method_test/configs/*.yaml` 약 30개에 asset pool 경로가 원본 절대경로로 하드코딩되어 있다.
  ```yaml
  root: "/home/yeocy/robotics/LLMforMani/Simulation/real2sim2real_pipeline/our_method_test/asset_pools/..."
  ```
  현재는 정상 동작하지만, 원본과 완전히 분리하려면 이 경로를 `airlab_twin` 기준으로 바꿔야 한다.
- `examples/1_collect_demos.py`의 예시 주석에도 원본 절대경로가 남아 있다.
- 일부 config가 참조하는 입력 폴더(`inputs_kist_cam2`, `inputs_kist_h3`, `inputs_kist_ramen`, `inputs_kist_front_right` 등)는 **원본에도 존재하지 않는다**. 해당 config는 원본에서도 실행되지 않던 상태다.
