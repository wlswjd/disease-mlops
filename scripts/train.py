import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import bentoml
import mlflow
import mlflow.sklearn

# ✅ MLflow 서버 주소 설정
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("DiseasePrediction")

# ✅ 절대경로 기반 CSV 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))  # /app/scripts
project_root = os.path.dirname(current_dir)               # /app
csv_path = os.path.join(project_root, "data", "raw", "Training.csv")

# 1. 데이터 로드
df = pd.read_csv(csv_path)

X = df.drop(columns=["prognosis"])
y = df["prognosis"]

# ✅ 결측값 처리 (실무적 방식)
imputer = SimpleImputer(strategy="most_frequent")  # 또는 strategy="mean"
X_imputed = imputer.fit_transform(X)

# 학습/검증 분리
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# ✅ MLflow run 시작
with mlflow.start_run():

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # ✅ MLflow 기록
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("train_accuracy", model.score(X_train, y_train))
    mlflow.log_metric("test_accuracy", model.score(X_test, y_test))

    # ✅ 모델 등록 (MLflow + BentoML)
    mlflow.sklearn.log_model(model, "model")
    bentoml.sklearn.save_model("disease_model", model)

    print("✅ 모델 학습 및 저장 완료 (MLflow + BentoML)")

