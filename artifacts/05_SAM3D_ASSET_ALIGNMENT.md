# 섹션 5: SAM3D 에셋의 방향(yaw)과 크기를 이미지에 맞추기

> 먼저 [00_COMMON.md](00_COMMON.md)를 읽는다. 이 섹션은 SAM3D 생성 모드(Step 1.5)가 만든 에셋이 입력 이미지와 **방향**과 **크기**가 안 맞는 문제를 다룬다.
>
> **원칙: 특정 물체나 태스크에만 맞춘 규칙(머그면 손잡이를, 접시면 높이를 …)은 넣지 않는다.** 어떤 물체가 들어와도 같은 방법으로 동작해야 한다. 판단이 필요하면 사람 대신 API(VLM)를 쓴다.

## 1. 현재 파이프라인 (SAM3D 생성 모드)

| 단계 | 하는 일 | 코드 |
|---|---|---|
| Step 1 | GPT 물체 캡션 + 마스크 | `our_method/pipeline/extraction.py` |
| Step 1.5 | GPT 필터(keep/discard, 테이블 D0·배경 D1)로 물체를 고르고, keep 된 물체마다 Step 1 마스크로 SAM3D → 풀 생성 | `pipeline/scene_object_filter.py`, `pipeline/sam3d_asset_generation.py`, `our_method_test/sam3d_asset_builder.py` |
| (빌더 mesh 단계) | GLB 세우기(y-up→z-up) → **실치수 스케일** → 감축 → UV + 텍스처 굽기 | `sam3d_asset_builder.py: stage_mesh` |
| (빌더 align 단계) | **에셋 생성의 일부.** yaw(z_angle)는 입력 카메라 render-and-compare, **크기는 depth 점군 3D 정합으로 메시 x/y/z 축별 배율**을 정해 메시에 굽는다. 배치용 `registration.pose_cam` 저장 | `utils/sam3d_pose_alignment.py`, `sam3d_asset_builder.py: stage_align` |
| Step 2 | 풀 렌더 100장과 비교해 모델과 yaw 선택 (정렬 pose 가 있는 검출은 Step 3 에서 무시된다) | `pipeline/matching.py` |
| Step 3 | 정렬 pose 가 있으면 자기 에셋을 그 pose(위치 + yaw, 바닥은 받침면)로 놓기만 한다. 없으면 점군 bbox 중심 + Step 2 yaw. 크기는 바꾸지 않는다(에셋 생성 때 구워짐) | `utils/scene_utils.py: align_model_pose(pose_cam=...)`, `pipeline/real_scene_generation.py` |

- 실행: `python our_models_task_generation.py --config configs/libero10_04.yaml`
- 결과 예시: `our_method_test/libero_10_sam3d/04_LIVING_ROOM_SCENE5_.../`. 비교 그림은 `robot_extrinsic/compare.png`(LIBERO 원본 / twin+로봇 / 50% 겹침)이다.

## 2. 문제

LIBERO-10 task 04(머그 3, 접시 2)에서 겹친 그림(`compare.png`)을 보면 다음이 보인다.

1. **방향(yaw)이 틀린다.** 빨간 머그는 손잡이가 있지만 yaw 가 틀려서 손잡이가 몸통 뒤로 숨었다. 노란 머그와 흰 머그는 손잡이 쪽이 대체로 맞다.
2. **크기가 틀린다.** 컵 몸통 지름이 원본보다 좁다. 높이는 거의 맞는다.
3. (부수) 위치도 조금 어긋난다. 흰 머그와 왼쪽 접시가 1~2 cm 어긋났다.

## 3. 원인

### 3.1 크기: 두 숫자로만 맞춘다

`sam3d_asset_builder.py: measure()` 는 마스크 영역 depth 를 월드 점군으로 바꿔 두 숫자만 잰다.

- 높이 = 물체 점 최고 z − 마스크 바로 바깥 링의 z 중앙값(받침면)
- 지름 = 위에서 본 물체 점의 최대 거리 (`xy_diameter`, convex hull 지름)

그다음 메시를 z 는 높이에, xy 는 지름에 맞춘다(`size_from: both`). 문제는 지름이 **손잡이를 포함한 최대 거리**라는 점이다.

- 손잡이가 가려지면 관측 지름이 작게 나온다 → 몸통이 줄어든다.
- SAM3D 가 손잡이를 길게 만들면 메시 지름에서 손잡이 비중이 커진다 → 같은 지름에 맞추면 몸통이 줄어든다.

| 물체 | 관측 지름 | LIBERO 정답 지름(손잡이 포함) | 관측 높이 | 정답 높이 |
|---|---|---|---|---|
| 흰 머그 | 0.124 | 0.131 | 0.124 | 0.115 (+1 cm 떠 있음) |
| 빨간 머그 | 0.120 | 0.126 | 0.148 | 0.135 (+1 cm) |
| 노란 머그 | 0.133 | 0.138 | 0.116 | 0.105 (+1 cm) |
| 접시 | 0.133 | 0.137 | (0.030) | 0.017 |

- 높이가 1 cm 큰 것은 측정 오류가 아니다. **LIBERO 메시가 테이블 위에 약 1 cm 떠 있어서** 생긴다(정답 윗면 z 와 관측 윗면 z 는 0.5 mm 이내로 일치한다). 실제 환경 입력에서는 생기지 않는다.

### 3.2 방향: Step 2 의 yaw 선택

Step 2 는 풀 렌더 100장(yaw 3.6° 간격) 중 입력 크롭과 비슷한 것을 DINO + GPT 로 고른다. 원통형 물체는 손잡이 말고는 yaw 단서가 거의 없어서 손잡이가 작거나 가려지면 틀린다. Step 3 의 `refine_yaw` 는 점군의 2D oriented bounds 로 yaw 를 보정하는데, 원형 단면에서는 이것도 정보가 없다.

### 3.3 위치: 원점과 배치 기준

- 에셋 원점은 메시 AABB 중심이다(`build_og_usd.build_usd` 가 AABB 중심으로 옮긴다). 손잡이가 있으면 AABB 중심이 몸통 중심에서 벗어난다.
- Step 3 은 "관측 점군 bbox 중심 = 에셋 bbox 중심" 으로 놓는다. 관측 점군은 카메라 쪽 면만 있어서 bbox 중심이 카메라 쪽으로 치우친다. yaw 가 틀리면 AABB 중심이 다른 곳으로 가서 몸통도 같이 밀린다.

## 4. 구현됨: 에셋 크기(depth x/y/z) + 배치 yaw (기본 켜짐)

**크기는 배치(Step 3)가 아니라 에셋을 만들 때(Step 1.5 빌더 `align` 단계) 정한다.** 정한 배율을 npz 메시(시각/충돌)에 굽고 그 크기로 USD 와 retrieval 렌더를 만든다. Step 3 은 크기를 바꾸지 않고 `pose_cam` 의 위치와 yaw 로 놓기만 한다. 물체 종류 규칙은 없다. `registration.method = yaw_render_compare+scale_xyz_depth_registration` (바뀌면 캐시된 풀도 `align,usd,render,overview` 를 다시 돈다).

- **yaw (1차):** 구운 텍스처 메시(`_build/<model>.npz`)를 **입력 카메라 그대로** open3d 레이캐스팅(CPU, 마스크 주변 크롭). 비용 `(1 − 실루엣 IoU) + 0.5·깊이 차(2 cm 상한) + 0.5·색 차`, 다른 물체에 가려진 픽셀(관측이 1 cm 이상 앞, 마스크 밖)은 뺀다. yaw 72개(5°)마다 위치를 "보이는 면 중심 = 관측 점 중심" 으로 맞추고, 25° 이상 떨어진 극소 상위 3개를 미세 조정한다. 비용 곡선 폭 < 0.03 이면 `symmetric`, 1·2위 차 < 0.02 면 `ambiguous`.
- **크기 (2차, depth 점군 3D 정합):** yaw 상위 후보(대칭이면 1개, 아니면 2개)에서 `(yaw, log sx, log sy, log sz, x, y, 바닥 z)` 를 Nelder-Mead 로 맞춘다. 배율은 **메시 자기 축(가로 x, 세로 y, 높이 z)마다 따로**다.
  - 비용: 마스크 안쪽 depth 점군 ↔ 입력 카메라에서 보이는 메시 표면 점(레이캐스팅), 양방향 최근접 거리 평균(1.5 cm 에서 자름). 관측을 못 덮으면(작음), 관측 없는 곳으로 튀어나오면(큼) 커진다. 가려진 렌더 점은 뺀다.
  - 사전: `0.2·[(lx−ly)² + (lz−(lx+ly)/2)²]` (l = SAM3D 원래 비율 기준 log 배율). 한 시점에서 거의 안 보이는 축만 붙잡는 약한 사전.
  - 제약: yaw 는 후보 ±20°, 배율 ±50%, 바닥 z 는 받침면 ±2 cm.
- **배치 pose:** 정합 위치와 yaw, 바닥은 받침면. 정합이 찾은 바닥 높이는 `registration.bottom_offset` 에 기록만 한다.
- 배율은 npz 에 굽고(`align_scale` 저장, 재실행 시 먼저 되돌린다) 그 뒤 `usd`, `render` 를 다시 돈다.
- 확인 그림: `<pool>/alignment.png` (정보 | 입력 크롭 | 정렬 전 겹침(yaw 만, mesh 단계 크기) | 정렬 후 렌더 | 정렬 후 겹침+마스크 윤곽).
- Step 3: `asset_pool.register_sam3d_poses` 가 **검출 이름**(`detection`)으로 pose 를 등록하고, `real_scene_generation` 이 그 검출의 cousin 을 자기 에셋으로 바꾼 뒤 `align_model_pose(pose_cam=...)` 로 `T_og_cam @ pose_cam` 위치와 yaw 에 놓는다(기울기는 OG 바닥에 맞춰 버린다). 그 뒤 `physics_settle` 은 그대로.
- 시도했다 버린 것 (04, 겹침 그림 기준): (a) 실루엣 비용으로 xy 등방 + z 배율, 바닥을 받침면에 고정 → 깊이 차를 yaw/배율이 메워 흰 머그 yaw 가 80° 튐. (b) 윗면 유지하고 바닥까지 늘이기 → 접시 두께 1.5~1.7 배. (c) 물체별 바닥 오프셋의 중앙값(공통 바닥)으로 받침면을 보정 → 머그 테두리는 잘 겹쳤지만, 크기를 에셋 생성 때 depth 로 먼저 정하기로 하고 보류(4-2 의 2번).

04 결과 (`robot_extrinsic/compare.png`, `sam3d_pool/alignment.png`):

| 물체 | yaw | 배율 x / y / z | 크기 (mesh 단계 → 최종, m) | 정합 바닥 | IoU |
|---|---|---|---|---|---|
| 흰 머그 | −2.7° | 1.18 / 1.14 / 0.93 | 0.080×0.124×0.124 → 0.094×0.142×0.115 | +0.6 cm | 0.866 |
| 빨간 머그 | −176.5° | 1.10 / 1.06 / 0.97 | 0.078×0.120×0.148 → 0.086×0.127×0.143 | +0.7 cm | 0.898 |
| 노란·흰 머그 | +178.4° | 1.07 / 1.08 / **0.83** | 0.091×0.133×0.116 → 0.098×0.143×**0.096** | **+2.0 cm (상한)** | 0.892 |
| 접시 0 | (대칭) | 1.06 / 1.03 / **0.72** | 0.131×0.132×0.023 → 0.138×0.137×0.017 | +1.5 cm | 0.981 |
| 접시 1 | (대칭) | 1.04 / 1.03 / **0.75** | 0.132×0.133×0.019 → 0.137×0.136×0.014 | +1.6 cm | 0.966 |

겹침 그림(사람 눈 확인): 가로·세로 폭과 손잡이 방향은 맞는다. 흰 머그·빨간 머그는 테두리도 거의 겹친다. **노란·흰 머그는 높이가 줄고(×0.83) 받침면에 내려놓여 테두리가 원본보다 낮게, 이중으로 보인다.** 이 장면의 물체가 추정 받침면보다 1~2 cm 떠 있는 입력이라, 물체별로 바닥을 풀면 바닥 높이와 z 배율이 맞바뀐다(접시도 z 0.72~0.75). 흰 머그 손잡이는 원본보다 크다(SAM3D 메시 모양). GPT 판정은 아직 없다.

### 구현 현황 (5·6·7절 대비)

✅ 완료 · 🟡 일부 · ⬜ 안 함

| 항목 | 상태 | 내용 |
|---|---|---|
| 5A SAM3D 자체 pose/scale 출력 | ⬜ | 드라이버는 아직 `gs`/`glb` 만 저장. 원격 `notebook/inference.py` 의 출력에 `rotation`/`translation`/`scale`(물체→카메라)이 있는 것은 확인함 |
| 5B 가림 반영 보이는 면 | ✅ | yaw 와 크기 정합 모두 레이캐스팅 렌더로 비교 |
| 5B 크기 정합 (x/y/z 축별, 비등방) | ✅ | depth 점군 3D 정합, SAM3D 비율 사전. 에셋 생성 때 메시에 굽는다 |
| 5B 바닥 높이 vs z 배율 분리 | ⬜ | 떠 있는 입력에서 높이가 줄어든다(노란 머그, 접시). 4-2 의 2번 |
| 5B 양방향 거리 | ✅ | 크기 정합에서 관측↔보이는 메시 표면 양방향 최근접 거리 |
| 5B yaw 다중 해 | 🟡 | 상위 후보 3개와 `ambiguous`/`symmetric` 표시를 저장. 고르는 것은 비용 최소값 |
| 5C VLM(GPT) 선택/검증 | ⬜ | GPT 호출 없음. 결과 확인은 사람 눈으로만 함 |
| 5D 정렬 pose 로 직접 배치 | ✅ | `align_model_pose(pose_cam=...)`, `T_og_cam @ pose_cam`, yaw 만 남기고 바닥에 세움 |
| 6절 연결 1~5 | ✅ | `register_mesh` 대신 새 `stage_align` 으로 연결 (6절 표시 참고) |
| 7절 평가 | 🟡 | 04 한 태스크, `compare.png` 정성 확인 + 정렬 IoU |

## 4-2. 남은 작업 (우선순위 순)

목표: **로봇을 넣어 겹친 `compare.png` 에서 컵의 회전과 크기가 원본과 맞는 것.** 판정은 사람 눈이 아니라 GPT API 로 한다. LIBERO 정답 pose 와 치수는 쓰지 않는다(실제 환경에는 없다).

1. **[5A] SAM3D 출력 pose/scale 저장과 사용**
   - `Kist/Kist/SAM3D/sam3d_driver.py`: `out["rotation"]`, `out["translation"]`, `out["scale"]` 을 `assets.json` 에 저장한다(로컬 파일만 고치면 된다. 실행 때 rsync 됨).
   - 규약 확인: GLB 로컬 프레임과 gaussian 로컬 프레임이 같은지, 카메라 규약(PyTorch3D 계열인지)을 `make_scene`, `SceneVisualizer.object_pointcloud` 에서 확인한다.
   - 04 를 `--force-sam3d` 로 다시 생성한다(원격 GPU 6, 7). 같은 seed 면 메시도 같아야 한다. 메시가 바뀌면 mesh 단계부터 다시 돈다.
   - 이 pose 를 입력 월드로 옮겨 yaw 와 크기(스케일 비)의 **초기값**으로 쓴다.
2. **[5B] 바닥 높이와 z 배율 분리** — 떠 있는 입력(받침면 추정 오차)에서 바닥이 올라간 만큼 z 배율이 줄어드는 문제.
   - 후보: 받침면 위 물체들의 바닥 오프셋 중앙값(공통 바닥)으로 바닥을 고정하고 크기를 다시 맞추고, Step 3 받침면을 그만큼 보정한다. 이전에 시도해 머그 테두리가 잘 겹쳤다(4절 (c)). 받침면이 여럿인 씬은 support_z 별로 묶어야 한다.
   - 손잡이처럼 SAM3D 가 크게 만든 부분은 축별 배율로도 못 고친다(흰 머그). 부분별 배율은 물체 규칙이 되므로 하지 않고, 1번(SAM3D 자체 scale)과 3번(GPT 후보 선택)으로 본다.
3. **[5C] GPT 로 선택과 검증**
   - 선택: `symmetric` 이 아닌 물체는 상위 후보(yaw, 스케일) 렌더 + 입력 크롭을 GPT 에 보여주고 고르게 한다. 프롬프트 고정, Responses API + JSON (`payload_filter_scene_objects` 형식).
   - 검증: 최종 `compare.png`(또는 물체별 크롭 쌍)를 GPT 에 주고 물체마다 회전, 크기, 위치가 맞는지 판정받아 `alignment_check.json` 으로 남긴다. 실패 판정이면 다음 후보로 다시 배치한다.
4. **[파이프라인] `compare.png` 자동 생성**
   - 지금은 `place_robot_from_extrinsic.py` 를 손으로 돌린다. 입력은 `robot_extrinsic/libero_robot_pose.json`, `libero_with_robot.png` 로 결과 폴더에 옮겨 두었다. LIBERO 입력 추출(`extract_libero10_inputs.py`) 때 이 둘도 입력 폴더에 저장하고, Step 3 뒤에 자동으로 돌게 한다.
5. **[7절] 다른 태스크로 일반성 확인**
   - 06(머그/접시/푸딩), 00·01·07(바구니 + 캔/박스)을 같은 설정으로 돌리고 GPT 판정과 실루엣 IoU 를 표로 남긴다.

## 4-1. 이전 실험: 메시 정합 (기본 꺼짐)

`sam3d_asset_builder.py: register_mesh()` 가 들어 있다. spec 의 asset 에 `register: true` 를 줄 때만 쓰이고, **Step 3 에는 아직 연결되지 않았다.**

- 변수: yaw, xy 스케일(등방), z 스케일, xy 위치. 바닥 z 는 받침면에 고정한다.
- 비용:
  - (a) 관측 점 → 카메라를 향한 메시 표면 거리(3 mm 단위, 3 cm 에서 자름)
  - (b) 실루엣 불일치: 보이는 표면을 투영했을 때 마스크 밖으로 나간 비율 + 마스크를 못 덮은 비율, 각각 가중치 4
- 탐색: yaw 36개 시작점 중 상위 4개에서 Powell.
- 결과는 `sam3d_assets.json` 의 `registration` 에 남긴다: `pose_world`, `pose_cam`(입력 카메라 opengl 기준 에셋 원점 pose).

오프라인 테스트 결과(04 의 SAM3D 메시, 물체당 약 30초):

| 물체 | yaw | 정합 지름 | 정합 높이 | 정답 지름 | 판단 |
|---|---|---|---|---|---|
| 흰 머그 | −41.7° | 0.144 | 0.126 | 0.131 | 지름이 너무 커짐 |
| 빨간 머그 | −118.7° | 0.131 | 0.152 | 0.126 | 지름 개선(0.120 → 0.131) |
| 노란 머그 | 179.9° | 0.137 | 0.119 | 0.138 | 거의 맞음 |
| 접시 0 | 99.1° | 0.159 | **0.073** | 0.137 | z 스케일 발산 |
| 접시 1 | −164.9° | 0.132 | 0.036 | 0.137 | z 커짐 |

관찰:

- **납작한 물체는 z 스케일이 정해지지 않는다.** 위에서 보면 높이 단서가 거의 없어서, 실루엣을 채우려고 z 를 키운다.
- 지름이 정답보다 커지는 경우가 있다. 실루엣 "못 덮은 비율" 항이 커지는 쪽으로 끌거나, 보이는 면 판정(법선만 사용, 가림 미반영)이 부정확해서로 보인다.
- 원통형은 yaw 의 비용 곡면이 평평하다. 손잡이 쪽 실루엣 차이만이 단서다.

## 5. 일반적인 해결 방향 (후보)

특정 물체 규칙 없이 적용 가능한 것만 적는다. 조합해서 쓰는 것을 권한다.

### A. SAM 3D Objects 의 자체 pose / scale 출력 쓰기 (먼저 확인할 것) — ⬜ 안 함 (출력 키 존재만 확인)

SAM 3D Objects 는 메시와 함께 **카메라 기준 물체 배치(회전, 이동, 스케일)** 를 추정하는 모델이다. 지금 `Kist/Kist/SAM3D/sam3d_driver.py` 는 `out["gs"]`(splat)와 `out["glb"]` 만 저장하고 나머지는 버린다.

- 할 일: 드라이버에서 `out.keys()` 를 출력해 pose / scale 관련 키(예: rotation, translation, scale, layout)가 있는지 확인하고, 있으면 `assets.json` 에 같이 저장한다.
- 있으면 이것을 정합의 **초기값**으로 쓴다. yaw 와 크기의 대부분이 해결되고, 정합은 미세 조정만 하면 된다. 단일 이미지 입력의 메시 좌표계와 pose 가 같은 모델에서 나오므로 가장 일관적이다.
- 이 드라이버는 원격 서버에 업로드되어 실행된다(`sam3d_from_mask.sh` 가 매번 rsync 한다). 로컬 파일만 고치면 된다.

### B. 기하 정합 개선 — ✅ yaw/크기/위치 (`stage_align`), 비등방 xy ⬜

- **스케일 사전(prior):** SAM3D 메시의 가로:세로:높이 비율을 기본으로 두고, 관측이 뒷받침할 때만 벗어나게 정규화한다. 예: `λ·(log(sz/sxy) − log(r_sam3d))²`. 납작한 물체의 z 발산이 이것으로 막힌다. 물체 종류 규칙이 아니라 "관측 정보가 부족한 축은 SAM3D 비율을 따른다" 는 일반 규칙이다.
- **보이는 면 판정:** 법선 대신 z-buffer 로 가림까지 반영한다(카메라 기준으로 메시를 래스터화). pyrender / nvdiffrast / open3d raycasting 중 하나를 쓴다.
- **양방향 거리:** 관측→메시 만이 아니라 "보이는 메시 → 관측" 도 넣어서, 관측이 없는 곳으로 메시가 튀어나오는 것을 막는다.
- **비등방 xy:** 상자처럼 가로세로가 다른 물체를 위해 sx ≠ sy 를 허용한다. yaw 가 정해진 뒤에만 의미가 있다.
- **yaw 다중 해:** 비용 상위 k 개 yaw 를 남기고 C 의 VLM 검증으로 고른다.

### C. VLM 으로 yaw 와 모양 검증 (판단이 필요한 부분) — ⬜ 안 함

기하로 구분이 약한 경우(원통형의 손잡이 방향, 대칭 물체)에만 쓴다.

- 상위 k 개 yaw 후보로 에셋을 **입력 카메라 시점에서** 렌더해, 입력 크롭과 나란히 VLM 에 보여주고 가장 일치하는 것을 고르게 한다. 프롬프트는 고정한다.
- 같은 방식으로 "가로/세로/높이 비율이 크롭과 다르다" 를 VLM 이 정성 판단하게 하고, 보정 방향만 받아 기하 정합에 제약으로 넣는 것도 가능하다. 숫자는 기하로, 선택과 검증은 VLM 으로 나눈다.
- 기존 GPT 호출 형식은 `our_method/models/gpt.py: payload_filter_scene_objects` 를 참고한다(Responses API, 이미지 여러 장 + JSON 출력).

### D. 원점과 배치 — ✅ 완료

- 에셋 원점은 메시 AABB 중심으로 둔다(요청 사항, 현재 그대로).
- 대신 Step 3 은 **정합 pose(`pose_cam`)로 바로 배치**한다. 점군 bbox 중심 배치와 Step 2 yaw 를 쓰지 않는다. 그러면 원점이 몸통 중심이 아니어도 위치가 맞는다.
- 변환: `T_scene_obj = T_scene_cam @ pose_cam` (`T_scene_cam` = `scene_0_info.json` 의 `cam_pose`, opengl 규약). 로봇 배치(`place_robot_from_extrinsic.py`)와 같은 방식이고, 로봇은 이 방식으로 원본과 거의 정확히 겹쳤다.

## 6. 연결 지점 (구현할 곳)

1. ✅ `sam3d_asset_builder.py: stage_align`: 정합 배율을 npz 메시에 굽고 `registration.pose_cam` 을 저장한다.
2. ✅ `pipeline/sam3d_asset_generation.py`: asset 에 `detection`(Step 1 검출 이름)을 넣는다. 캐시된 풀에 `registration` 이 없으면 `align` 만 돈다.
3. ✅ `utils/asset_pool.py`: `register_sam3d_poses` / `sam3d_pose` 가 `pose_cam` 을 **검출 이름 기준**으로 등록한다(04 에서 접시 2개가 둘 다 `smallplate1` 로 매칭되던 문제 해결).
4. ✅ cousin 고정: `gaia.py` 대신 `pipeline/real_scene_generation.py` 의 물체 루프에서 정렬 pose 가 있는 검출은 cousin 을 **자기 에셋**으로 바꾼다.
5. ✅ `utils/scene_utils.py: align_model_pose(pose_cam=...)`: 그 pose 로 바로 놓고(스케일 그대로, `tf_from_cam` 계산) 반환한다. 그 뒤 `physics_settle` 은 그대로. 받침면 높이 보정은 `real_scene_generation._load_and_setup_environment` 가 카메라를 내려서 한다.

## 7. 평가 (일반성 확인)

**LIBERO 정답 pose/치수는 쓰지 않는다.** 실제 환경 입력에는 없으므로 이미지(+깊이, 카메라)만으로 확인한다. 3.1 의 정답 치수 표는 원인 분석 참고용이고, 파이프라인과 평가에는 쓰지 않는다. 여러 태스크에 같은 설정으로 돌려서 본다.

- **주 판정: 로봇 겹침 그림.** `place_robot_from_extrinsic.py` 의 `robot_extrinsic/compare.png` (원본 / twin+로봇 / 50% 겹침). 물체마다 회전, 크기, 위치가 맞는지를 **GPT 로 판정**해 json 으로 남긴다(⬜, 4-2 의 3번). 사람 눈 확인은 보조로만 쓴다.
- **수치: 실루엣 IoU.** 입력 카메라 렌더의 물체 마스크 vs 입력 마스크(`alignment.png` 의 IoU, 그리고 twin 렌더 기준). 정답 없이 계산된다.
- **기록:** 태스크 × 물체 표(GPT 판정, IoU, symmetric/ambiguous)를 이 문서에 남긴다.

관절 없는 물체만 있는 태스크: 04(머그/접시), 06(머그/접시/푸딩), 00·01·07(바구니 + 캔/박스, 바구니는 오목해서 충돌 메시 주의).

## 8. 재현과 주의 사항

- **API 키:** `real2sim2real_pipeline/execute` 의 `export OPENAI_API_KEY=...` 줄을 쓴다. `chatgpt_api_key` 파일은 형식이 달라 `Connection error.` 가 난다.
- **SAM3D 원격 서버:** 빈 GPU 를 config `SAM3DAssetGenerator.call.gpus` 로 준다. `nvidia-smi` 로 먼저 확인한다.
- **Step 1 마스크:** `*_nonprojected_mask_pruned.png` 는 픽셀을 솎아낸 점무늬다. SAM3D 입력이나 치수 측정에는 `*_nonprojected_mask.png` 를 쓴다.
- **import 순서:** `og.launch()` 뒤에 `our_method` / torchvision / PIL 을 처음 import 하면 Isaac 번들 PIL 과 충돌해 segfault 가 난다. launch 전에 import 한다.
- **실행 경로:** 스크립트는 절대경로로 실행한다. 작업 폴더가 바뀌어 `can't open file` 이 난 적이 있다.
- **Isaac Sim 멈춤:** 가끔 확장 기능 로딩 중에 멈춘다. 로그가 수 분간 안 늘면 PID 로 종료하고 다시 돌린다.
- **로봇 겹침 그림 다시 만들기:** `python place_robot_from_extrinsic.py --scene_info <save_dir>/step_3_output/scene_0/scene_0_info.json --robot_pose <save_dir>/robot_extrinsic/libero_robot_pose.json --camera_info <save_dir>/inputs/camera_info.json --pool <save_dir>/sam3d_pool --reference <save_dir>/robot_extrinsic/libero_with_robot.png --out <save_dir>/robot_extrinsic` (acdc env, 절대경로, `PYTHONPATH` 에 airlab_twin).
- **Step 3 만 다시 돌리기:** `configs/libero10_04_s3.yaml`. Step 1.5 는 필터 판단과 풀을 캐시로 재사용한다.
- **풀 이식성:** 빌더는 평문 `<model>.usd`(텍스처 `./textures/` 상대경로)를 같이 만들고, `use_pool_usd` 가 그것을 제자리에서 연다. 풀 폴더를 옮겨도 된다.
