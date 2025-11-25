import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2S 
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import load_model 
from tensorflow.keras import mixed_precision
from sklearn.utils import class_weight 
from PIL import ImageFile
import sklearn

ImageFile.LOAD_TRUNCATED_IMAGES = True
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

BATCH_SIZE = 32 
IMG_SIZE = (224, 224)
NUM_CLASSES = 7
DATA_DIR = "/home/yeonji/projects/opensource/DATA"

# Mixup 
class MixupGenerator(tf.keras.utils.Sequence):
    def __init__(self, generator, alpha=0.2):
        self.generator = generator
        self.alpha = alpha
        self.batch_size = generator.batch_size
        
    def __len__(self):
        return len(self.generator)
    
    def __getitem__(self, index):
        X, y = self.generator[index]
        batch_size = X.shape[0]
        
        # 랜덤 선택
        indices = np.random.permutation(batch_size)
        
        # 섞을 비율 
        lam = np.random.beta(self.alpha, self.alpha, batch_size)
        lam = lam.reshape(-1, 1, 1, 1)
        
        # 이미지 섞기
        X_mix = X * lam + X[indices] * (1 - lam)
        
        # 라벨 섞기
        lam_label = lam.reshape(-1, 1)
        y_mix = y * lam_label + y[indices] * (1 - lam_label)
        
        return X_mix, y_mix

# 데이터 로드 
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,     
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator_raw = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

train_classes = train_generator_raw.classes
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_classes),
    y=train_classes
)

class_weight_dict = dict(enumerate(class_weights))

print(f"분노 가중치 : {class_weight_dict}")

train_generator = MixupGenerator(train_generator_raw, alpha=0.2)

val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)
# EfficientNetB0
# EfficientNetV2-S
# ResNet50V2
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)

# Head 
x = Dense(512)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.4)(x)

x = Dense(256)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.3)(x) 
prediction = Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)
#model = load_model("best_model_v2s_mixup.h5") 
model = Model(inputs=base_model.input, outputs=prediction)

# 학습 설정 
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1) 
model.compile(optimizer="adam", loss=loss_fn, metrics=['accuracy']) 
#옵티마이저 adam에서 resume하고 바꿈

checkpoint = ModelCheckpoint(
    'best_model_b0_mixup_v2.h5', 
    monitor='val_accuracy', 
    save_best_only=True, 
    verbose=1
)
early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True) 
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6,verbose=1)

print("\n 1차 학습 시작")
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    callbacks=[checkpoint, early_stop, reduce_lr],
    class_weight=class_weight_dict
)

print("\n 2차 학습 시작")
base_model.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5), # 아주 낮게
    loss=loss_fn, 
    metrics=['accuracy']
)

history_fine = model.fit(
    train_generator,
    epochs=60, 
    validation_data=val_generator,
    callbacks=[checkpoint, early_stop, reduce_lr],
    class_weight=class_weight_dict
) 

print("\n학습 완료!")