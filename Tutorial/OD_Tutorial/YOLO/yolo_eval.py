from ultralytics import YOLO


class YoloEval:
    def __init__(self, model_path):
        self.model = YOLO(model_path) #훈련이 완료된 모델 불러오기 

    def eval(self, image_path, conf = 0.25, img_save = False): #conf = 0.25이상인 것만 출력 
        results = self.model.predict(source=image_path, conf=conf) #이미지 예측 수행
        for i, result in enumerate(results):
            if img_save:
                result.save(filename=f'result_{i}.jpg')  # 결과 이미지 저장


if __name__ == "__main__":
    yolo_eval = YoloEval("runs/detect/yolo_custom/weights/best.pt")
    yolo_eval.eval(["sample_test.jpg"], img_save=True)