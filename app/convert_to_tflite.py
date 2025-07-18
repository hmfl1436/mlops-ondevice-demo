import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

# 1. 데이터 준비
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# 2. Keras 모델 정의 및 학습
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, verbose=0)
model.save("app/iris_keras_model.h5")

# 3. 평가 (선택)
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Keras 테스트 정확도: {acc:.2f}")

# 4. TFLite 변환 (INT8 양자화 옵션 포함)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("app/iris_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite 변환 완료: app/iris_model.tflite")
