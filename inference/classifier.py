import cv2
import numpy as np
import mediapipe as mp
import joblib
from tensorflow.keras.models import load_model
from mediapipe import solutions as mp_solutions

# 설정
MODEL_PATH = '/home/yeonji/projects/opensource/(best)mlp_v2.keras'
SCALER_PATH = '/home/yeonji/projects/opensource/(best)mlp_v2.pkl'
EMOTIONS = ['슬픔', '기쁨', '분노', '불안', '당황', '상처', '중립']

# 전역 변수 로드
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# MediaPipe 초기화
mp_face_detection = mp_solutions.face_detection.FaceDetection(
    min_detection_confidence=0.5
)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,     
    min_detection_confidence=0.5
)

def detect_face_region(img):
    """얼굴 bounding box를 MediaPipe Face Detection으로 찾기"""
    results = mp_face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not results.detections:
        return None

    detection = results.detections[0]
    box = detection.location_data.relative_bounding_box
    h, w, _ = img.shape

    x_min = int(box.xmin * w)
    y_min = int(box.ymin * h)
    x_max = int((box.xmin + box.width) * w)
    y_max = int((box.ymin + box.height) * h)

    return x_min, y_min, x_max, y_max


def preprocess_face(img, padding_ratio=0.3):
    """얼굴 bbox + padding 으로 crop"""
    bbox = detect_face_region(img)
    if not bbox:
        return None

    x_min, y_min, x_max, y_max = bbox

    pad_x = int((x_max - x_min) * padding_ratio)
    pad_y = int((y_max - y_min) * padding_ratio)

    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(img.shape[1], x_max + pad_x)
    y_max = min(img.shape[0], y_max + pad_y)

    cropped = img[y_min:y_max, x_min:x_max]
    if cropped.size == 0:
        return None
    return cropped


def predict_image(image_bytes):
    """
    Flask에서 받은 이미지 바이트를 입력받아 감정 문자열을 반환하는 함수
    """
    try:
        # OpenCV 이미지로 변환
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"status": "error", "message": "이미지 파일이 아닙니다."}

        # 얼굴 crop
        cropped_img = preprocess_face(img)
        if cropped_img is None:
            return {"status": "error", "message": "얼굴을 찾을 수 없습니다."}

        # Face Mesh로 랜드마크 추출
        results = face_mesh.process(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return {"status": "error", "message": "얼굴 랜드마크를 찾을 수 없습니다."}

        landmarks = results.multi_face_landmarks[0].landmark

        # flatten
        mesh_coords = []
        for lm in landmarks:
            mesh_coords.extend([lm.x, lm.y, lm.z])

        features = np.array(mesh_coords, dtype=np.float32).reshape(1, -1)

        # 입력 차원 체크
        if model.input_shape[1] != features.shape[1]:
            return {
                "status": "error",
                "message": f"입력 차원 불일치: model={model.input_shape[1]}, now={features.shape[1]}"
            }

        # StandardScaler 적용
        features_scaled = scaler.transform(features)

        # 모델 예측
        prediction = model.predict(features_scaled, verbose=0)
        label_idx = int(np.argmax(prediction, axis=1)[0])
        confidence = float(np.max(prediction))

        return {
            "status": "success",
            "emotion": EMOTIONS[label_idx],
            "confidence": f"{confidence * 100:.2f}%"
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

