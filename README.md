# OSS-CV
이 모델은 **MediaPipe FaceMesh (478개 랜드마크)** 데이터를 입력으로 받아  
**7가지 감정 클래스(슬픔, 기쁨, 분노, 불안, 당황, 상처, 중립)** 중 하나를 예측합니다.  
모델은 TensorFlow 기반 MLP이며, 입력 정규화를 위해 `StandardScaler`를 사용했습니다.

---

## 📁 구성 파일

| 파일명 | 설명 |
| `best_mlp.keras` | 학습 완료된 MLP 모델 (Keras native 포맷) |
| `scaler.pkl` | 입력 정규화용 StandardScaler 객체 |
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
