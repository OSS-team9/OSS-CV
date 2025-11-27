import os
import glob
import json
import unicodedata
import numpy as np
import cv2
import joblib
import mediapipe as mp
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 경로 설정
BASE_PATH = "/home/yeonji/projects/opensource"
MODEL_PATH = os.path.join(BASE_PATH, "best_mlp.keras")
SCALER_PATH = os.path.join(BASE_PATH, "scaler.pkl")
DATA_DIR = os.path.join(BASE_PATH, "DATA")
METADATA_DIR = "/home/yeonji/projects/opensource/LABEL"

EMOTION_CATEGORIES = ['슬픔', '기쁨', '분노', '불안', '당황', '상처', '중립']
PADDING_RATIO = 0.3
SEED = 42

def normalize_fname(name: str) -> str:
    return unicodedata.normalize("NFC", os.path.basename(name).strip()).lower()

def load_all_metadata(metadata_dir: str) -> dict:
    metadata_map = {}
    json_files = glob.glob(os.path.join(metadata_dir, '*.json'))
    
    print(f"JSON 메타데이터 로드... ({len(json_files)}개 파일)")
    
    for json_path in json_files:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for item in data:
            if 'filename' not in item: continue
            
            key = normalize_fname(item['filename'])
            
            if 'annot_A' in item and 'boxes' in item['annot_A']:
                metadata_map[key] = item['annot_A']['boxes']
            elif 'annot_B' in item and 'boxes' in item['annot_B']:
                metadata_map[key] = item['annot_B']['boxes']
            elif 'annot_C' in item and 'boxes' in item['annot_C']:
                metadata_map[key] = item['annot_C']['boxes']
    
    print(f"메타데이터 로드 완료: 총 {len(metadata_map)}개 이미지 정보")
    return metadata_map

def process_image(img_path: str, boxes: dict, face_mesh) -> list:
    img = cv2.imread(img_path)
    if img is None: return None

    h, w = img.shape[:2]
    
    # JSON 좌표 기반 패딩 및 크롭 영역 계산
    minX, minY, maxX, maxY = boxes['minX'], boxes['minY'], boxes['maxX'], boxes['maxY']
    pad_w = (maxX - minX) * PADDING_RATIO
    pad_h = (maxY - minY) * PADDING_RATIO
    
    crop_x1 = int(max(0, minX - pad_w))
    crop_y1 = int(max(0, minY - pad_h))
    crop_x2 = int(min(w, maxX + pad_w))
    crop_y2 = int(min(h, maxY + pad_h))
    
    cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]
    if cropped.size == 0: return None
    
    # 랜드마크 추출
    results = face_mesh.process(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    
    if not results.multi_face_landmarks:
        return None

    # (x, y, z) 좌표 평탄화 (Flatten)
    landmarks = []
    for lm in results.multi_face_landmarks[0].landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
        
    return landmarks

if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    metadata = load_all_metadata(METADATA_DIR)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
    )
    
    # MediaPipe 초기화
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

    X_total, y_total = [], []
    
    print("데이터 전처리 및 랜드마크 추출 시작...")

    for label_idx, emotion in enumerate(EMOTION_CATEGORIES):
        folder_path = os.path.join(DATA_DIR, emotion)
        if not os.path.exists(folder_path): continue
        
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        
        for fname in tqdm(files, desc=f"{emotion}"):
            key = normalize_fname(fname)
            
            if key in metadata:
                landmarks = process_image(os.path.join(folder_path, fname), metadata[key], face_mesh)
                if landmarks:
                    X_total.append(landmarks)
                    y_total.append(label_idx)

    if X_total:
        X_total = np.array(X_total)
        y_total = np.array(y_total)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_total, y_total, 
            test_size=0.2, 
            random_state=SEED, 
            stratify=y_total
        )

        print(f"검증 데이터: {len(X_val)}개")

        # 스케일링 적용
        X_val = scaler.transform(X_val)
        
        y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
        
        # 결과 출력
        print("\n" + "="*60)
        print(classification_report(y_val, y_pred, target_names=EMOTION_CATEGORIES, zero_division=0))
        print("="*60)

        # Confusion Matrix 시각화
        plt.rc('font', family='NanumGothic')
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix(y_val, y_pred), annot=True, fmt='d', cmap='Blues',
                    xticklabels=EMOTION_CATEGORIES, yticklabels=EMOTION_CATEGORIES)
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix_mlp.png')
        print("결과 저장 완료: confusion_matrix.png")
        
    else:
        print("유효한 데이터를 찾을 수 없습니다. 경로와 JSON 파일을 확인하세요.")