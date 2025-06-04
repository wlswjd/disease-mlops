# bento_service.py
import bentoml
from bentoml.io import JSON
import numpy as np
import pandas as pd

# 저장된 BentoML 모델 불러오기
model_ref = bentoml.sklearn.get("disease_model:latest")
model_runner = model_ref.to_runner()

from bentoml import Service

svc = Service("disease_service", runners=[model_runner])

# 입력 데이터를 JSON으로 받음
@svc.api(input=JSON(), output=JSON())
def predict(input_data):
    # 입력값을 DataFrame으로 변환 (컬럼 수에 따라 맞춤)
    input_df = pd.DataFrame([input_data])
    prediction = model_runner.run(input_df)
    return {"prediction": prediction[0]}

