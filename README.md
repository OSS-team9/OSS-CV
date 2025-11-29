# OSS-CV

이 모델은 **MediaPipe FaceMesh (478개 랜드마크)** 데이터를 입력으로 받아

**7가지 감정 클래스(슬픔, 기쁨, 분노, 불안, 당황, 상처, 중립)** 중 하나를 예측합니다.

모델은 TensorFlow 기반 MLP이며, 입력 정규화를 위해 `StandardScaler`를 사용했습니다.

---

## 📁 구성 파일

| 파일명 | 설명 |<br>

| `(best)mlp_v2.keras` | 학습 완료된 MLP 모델 (Keras native 포맷) | <br>

| `(best)mlp_v2.pkl` | 입력 정규화용 StandardScaler 객체 |<br>

| `label_map.json` | 라벨 인덱스 → 감정 이름 매핑 정보 |

---

## 🧠 모델 정보

- **Input shape:** `(478, 3)` → reshape 후 `(1, 1434)`
- **Output:** 7-class softmax 확률 벡터
- **Classes:**
0: 슬픔
1: 기쁨
2: 분노
3: 불안
4: 당황
5: 상처
6: 중립

---

## 🧩 전처리 파이프라인

1. 얼굴 이미지가 업로드
2. MediaPipe Face Detection으로 얼굴 bounding box 추출
3. bounding box 주변으로 padding_ratio=0.3 만큼 확장하여 crop
4. crop된 얼굴 이미지를 MediaPipe FaceMesh에 입력
5. 478개 얼굴 랜드마크 (x, y, z) → shape: (478, 3)
6. (478, 3)을 flatten하여 (1, 1434) 벡터로 변환
7. StandardScaler((best)mlp_v2.pkl)로 정규화
8. 정규화된 벡터를 MLP 모델((best)mlp_v2.keras)에 입력 → 7-class softmax 출력

---

### 😶 Medipipe  설정 값

※ 위 설정을 변경하면 입력 특징 분포가 달라져 모델 성능이 저하될 수 있습니다.

[ Face Detection ] 

min_detection_confidence = 0.5

[ FaceMesh]

static_image_mode=True,
max_num_faces=1,
refine_landmarks=True,

min_detection_confidence=0.5

---

project-root/
├─ model/ : 학습된 MLP 모델(.keras), StandardScaler(.pkl) 등 모델
│  ├─ (best)mlp_v2.keras
│  ├─ (best)mlp_v2.pkl
│  └─ ...
│
├─ label/ :감정 라벨 인덱스
│  ├─ labels_기쁨.npy
│  ├─ labels_당황.npy
│  └─ ...

├─ train/

│  ├─ EfficientNet.py # efficientnet 학습 코드 
│  └─ MLP.py  # mlp 학습 코드
│
├─ inference/
│  ├─ classifier.py  # 추론 모듈
│  └─ test_.py  # 로컬 테스트용 
│
├─ etc/          
│  ├─ MLP_preprocessing.py  # mlp 모델용 데이터 전처리 코드

│  ├─choose_bad_files.py  # 깨진 이미지 삭제하는 코드 

│  ├─choose_bad_files.py  # 이미지 기반 모델 validation 및 confusion matrix 그리는 코드
│  └─ visualize_mlp_matrix.py # MLP 모델 validation 및 confusion matrix 그리는 코드
│
└─ README.md
