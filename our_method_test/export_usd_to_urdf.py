# Isaac Sim의 Python API를 import
from omni.isaac.kit import SimulationApp
# Isaac Sim 애플리케이션 초기화 (headless 모드 옵션)
simulation_app = SimulationApp({"headless": False})

import omni.isaac.core.utils.extensions as extensions
import omni.isaac.urdf as urdf_extension


# USD to URDF exporter 확장 확인 및 활성화
if not extensions.is_extension_enabled("omni.isaac.urdf"):
    extensions.enable_extension("omni.isaac.urdf")

# 변환하려는 USD 파일 경로
usd_path = "<path/to/object.usd>"

# 변환된 URDF를 저장할 경로
urdf_output_path = "<path/to/object.urdf>"

# USD to URDF exporter 인터페이스 가져오기
urdf_interface = urdf_extension.acquire_urdf_interface()

# USD 파일을 URDF로 변환
success = urdf_interface.export_usd_to_urdf(
    usd_path=usd_path,
    urdf_path=urdf_output_path,
    merge_fixed_joints=True,  # 고정 조인트 병합 옵션
    export_meshes=True,       # 메시 내보내기 옵션
    mesh_format="obj"         # 메시 포맷 (obj, stl 등)
)

if success:
    print(f"성공적으로 USD 파일을 URDF로 변환했습니다: {urdf_output_path}")
else:
    print("USD 파일 변환 실패")

# 필요한 작업 후 Isaac Sim 애플리케이션 종료
simulation_app.close()