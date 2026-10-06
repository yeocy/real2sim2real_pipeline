# 섹션 2 — 가려진 부분의 depth/점군 채우기 (Occlusion Completion)

> 먼저 [00_COMMON.md](00_COMMON.md)를 읽을 것. 작업 폴더: `our_method_test/acdc_out_s2_depth` (`acdc_out_s5_physics`를 복사해서 시작)

## 목표
물체 배치(위치, yaw, 크기)는 물체별 점군의 정확도에 달려 있다. 지금은 **물체 마스크 × depth map**으로 물체별 점군을 만든다. 그래서 다른 물체에 가려진 부분은 점군에서 빠지고, 물체 중심과 extent가 치우친다.
**마스크와 depth를 보고 가려진 영역을 찾아서, 그 영역의 depth(또는 점군)를 채우는 모듈**을 만든다. 출력은 Step 3 배치가 그대로 쓸 수 있는 형태여야 한다.

## 섹션 1·5 이후 달라진 점
- 섹션 1이 `support_yaw_align`을 넣었다. 인덕션처럼 지지 물체 위의 직사각형 물체는 **외곽선(convex hull)** 으로 yaw와 footprint를 정한다. 인덕션 마스크에 뚫린 구멍(냄비가 가린 부분)은 yaw 문제에서는 이미 우회됐다. 기록: footprint 0.280×0.380 m = 에셋 실치수, fill 0.98
- 그래서 이 섹션의 우선 대상은 **외곽선으로도 해결되지 않는 경우**다.
  - 앞 테이블(`table_0`): 상판 위 물체들 때문에 구멍이 여러 개 뚫리고, 앞쪽 가장자리와 다리가 잘린다.
  - 뒤 받침대(`table_1`): 앞 테이블과 정수기에 가려 아래쪽과 다리가 빠진다.
  - **위치와 높이**: 구멍 난 점군의 중심과 z 범위가 치우치는 문제. `support_yaw_align`은 yaw만 고친다.
- 섹션 5가 `physics_settle`을 넣었다. 저장 직전에 관통을 풀고 중력으로 안착시키므로, **높이(z)** 오차는 상당 부분 사후 보정된다. 이 섹션은 그 단계가 고칠 수 없는 **xy 위치와 크기·yaw** 오차를 줄이는 데 집중한다. 효과를 볼 때는 `physics_settle`을 켠 결과끼리 비교한다.
- 점군 이상치 제거 옵션 `GAIA_PC_OUTLIER_FILTER=1`도 생겼다. 경계에서 배경이 섞인 점을 자르는 기능이라 이 섹션과 역할이 다르다. 둘을 같이 켜도 충돌하지 않게 만든다.

## 확인할 파일 (`acdc_out_s5_physics/step_1_output/`)
- `segmented_objects/<obj>_nonprojected_mask.png`, `<obj>_nonprojected_mask_pruned.png`
- `step_1_depth.png`: 16bit, `depth_limits=[0, 20]` m로 정규화. 원본은 `inputs_kist_twin_full/camera_depth.npy`
- `step_1_output_info.json`: K, `z_direction`(바닥 법선), `origin_pos`
- `step_3_output/scene_0/scene_0_info.json`: 물체별 `bbox_extent`, `tf_from_cam`, 인덕션의 `support_yaw_align` 기록

## 데이터 흐름 (어디에 끼울지)

```
camera_depth.npy (GT)
  └─ extraction.py:324  _estimate_depth                  → step_1_depth.png
  └─ extraction.py:388  compute_point_cloud_from_depth   → 전체 점군
  └─ extraction.py:849  denoise_obj_point_cloud + *_mask_pruned.png 저장
Step 3 (real_scene_generation.py):
  └─ :227  self.pc = compute_point_cloud_from_depth(depth=depth, K=self.K)
  └─ :403  _load_obj_pc(obj_name)   ← 물체 점군을 꺼내는 단일 지점. 여기에 연결
  └─ :432  _compute_footprints      ← support_yaw_align 이 쓰는 외곽선 (같은 점군 사용)
  └─ scene_utils.align_model_pose   ← 점군 min/max AABB로 위치·스케일 정렬
```

| 위치 | 내용 |
|---|---|
| `our_method/utils/processing_utils.py:226`, `:259` | `process_depth_linear` / `unprocess_depth_linear` |
| `processing_utils.py:285` `compute_point_cloud_from_depth` | 점군 생성 |
| `processing_utils.py:456` `denoise_obj_point_cloud` | 물체 점군 노이즈 제거 |

## 접근 아이디어 (정해진 답 아님)
1. **가림 관계 판정**: 두 마스크가 이웃하고, 경계에서 A의 depth가 B보다 작으면 A가 B를 가린다. 이렇게 물체 간 가림 그래프를 만든다.
2. **아모달 마스크**: 가려진 물체 B의 보이는 마스크와 가리는 물체 A의 마스크를 합쳐 B의 전체 윤곽을 추정한다(hole fill, convex hull, 평면 경계 외삽 등).
3. **depth 채우기**: B의 가려진 픽셀에 B의 보이는 표면을 평면 또는 저차 곡면으로 피팅해 외삽한다. 테이블 상판과 인덕션은 바닥 법선과 평행한 평면 피팅으로 충분하다.
4. **출력**: 물체별 `<obj>_completed_mask.png`와 `<obj>_completed_depth.npy`(또는 점군 `.npy`)를 `segmented_objects/`에 **추가**한다. 기존 파일은 덮어쓰지 않는다. `_load_obj_pc`에 "completed 파일이 있고 옵션이 켜져 있으면 그것을 쓴다"는 분기만 넣는다.
5. 그릇이나 냄비처럼 속이 빈 물체의 안쪽은 채우지 않는다. 평면형 지지물부터 적용하고, 대상 범위는 config로 제한한다.

## 완료 기준
- `table_0`, `table_1`, `induction_cooktop_0`에서 채운 마스크와 depth를 시각화해 저장한다(전/후 오버레이).
- 채운 점군의 AABB extent가 실제 에셋 치수에 가까워진다. 앞 테이블 1.2×0.8 m, 뒤 받침대 0.65×0.55 m, 인덕션 0.28×0.38 m 기준으로, 기존 대비 오차가 줄었음을 표로 보인다.
- 옵션을 켜고 Step 3 → 4·5 → 6 → 7을 다시 돌렸을 때, 씬(인덕션 정렬, 그릇 안 내용물 포함)이 현재 기준 결과보다 나빠지지 않는다.
- 옵션을 끄면 현재 기준 결과가 그대로 재현된다.

## 산출물
- 새 모듈(예: `our_method/utils/occlusion_completion.py`) + 연결 옵션 + config `configs/s2_depth.yaml`
- 전후 비교 이미지와 extent 오차 표
- 아래 "확정된 출력 형식"을 채운다.

---
### 확정된 출력 형식 (작업 세션이 채울 것)
- (미정)
