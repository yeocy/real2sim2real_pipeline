import cv2

# 이미지 읽기
image = cv2.imread('acdc_output/step_3_output/scene_0/scene_0_visualization.png')

# 이미지가 제대로 읽혔는지 확인
if image is None:
    print("이미지를 찾을 수 없습니다.")
else:
    height, width, _ = image.shape
    print("이미지 크기:", image.shape)  # (900, 1600, 3)

    # 오른쪽 절반만 자르기
    right_half = image[:, width // 2:]  # 전체 세로, 가로 800~1600

    # 이미지 저장
    cv2.imwrite('right_half.png', right_half)

    # 저장된 이미지 보기 (선택사항)
    cv2.imshow('Right Half', right_half)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
