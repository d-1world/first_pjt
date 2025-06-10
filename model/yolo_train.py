import ultralytics

model = ultralytics.YOLO("yolo11n.pt")

model.train(
    data="/home/rokey/ros2_ws/src/first_pjt/Tutorial/OD_Tutorial/datasets/data.yaml",  # 데이터 경로
    epochs=100,  # 에폭 수
    imgsz=640,  # 이미지 크기
    batch=16,  # 배치 크기
)