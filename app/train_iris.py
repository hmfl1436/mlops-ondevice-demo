from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. 데이터 로드
iris = load_iris()
X, y = iris.data, iris.target

# 2. train/test 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. MLP 분류기 학습
clf = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=300, random_state=42)
clf.fit(X_train, y_train)

# 4. 예측 및 평가
y_pred = clf.predict(X_test)
print("테스트 정확도: {:.2f}".format(accuracy_score(y_test, y_pred)))
print("분류 리포트:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# 5. 모델 저장
joblib.dump(clf, "iris_mlp.joblib")
print("모델이 /workspace/app/iris_mlp.joblib 로 저장되었습니다.")
