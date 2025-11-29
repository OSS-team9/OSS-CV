import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# 설정
MODEL_PATH = '/home/yeonji/best_model_resnet.h5' 
DATA_DIR = "/home/yeonji/projects/opensource/DATA"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 모델 로드 
print(f" 모델 불러오는 중: {MODEL_PATH}")
model = load_model(MODEL_PATH, compile=False)
print(" 모델 로드 완료")

# validation
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,validation_split=0.2)

val_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation', 
    shuffle=False        
)

# 예측
print(" 예측 수행 중...")
Y_pred = model.predict(val_generator, verbose=1)
y_pred = np.argmax(Y_pred, axis=1) 
y_true = val_generator.classes     

# 라벨 이름 불러오기
class_labels = list(val_generator.class_indices.keys()) 

print("\n" + "="*50)
print("분류 결과")
print("="*50)
print(classification_report(y_true, y_pred, target_names=class_labels))

# confusion matrix 
cm = confusion_matrix(y_true, y_pred)

# 그래프 그리기
plt.rc('font', family='NanumGothic') 
plt.rc('axes', unicode_minus=False)

plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_true, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_labels, yticklabels=class_labels)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

# 결과 저장
save_path = 'confusion_matrix.png'
plt.savefig(save_path)
print(f"\n저장 완료")
