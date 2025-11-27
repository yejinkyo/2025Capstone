import pandas as pd
import numpy as np
import shap
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# =====================
# 1. 데이터 전처리
# =====================
def preprocess_telco(df):
    df = df.copy()

    # TotalCharges → 숫자 변환
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    df = df.dropna(subset=["Total Charges"])

    target = "Churn Value"
    y = df[target]

    # 제외하지 않을 feature 목록
    service_features = [
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
        "Phone Service",
        "Multiple Lines",
        "Paperless Billing",
        "Internet Service",
        "Contract",
        "Payment Method",
    ]

    numeric_features = ["Monthly Charges", "Total Charges", "Tenure Months"]

    categorical_features = service_features

    X = df[categorical_features + numeric_features]

    # 원핫 인코더
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    return X, y, preprocessor, categorical_features, numeric_features

# ================
# 2. 모델 학습
# ================
def train_churn_model(df):
    X, y, preprocessor, cat_cols, num_cols = preprocess_telco(df)

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", LogisticRegression(max_iter=300)),
        ]
    )

    model.fit(X, y)

    return model, X, y, cat_cols, num_cols

# ======================
# 3. SHAP 계산
# ======================
def compute_shap_importance(model, X_sample):
    explainer = shap.Explainer(model.named_steps["clf"])
    shap_values = explainer(model.named_steps["preprocess"].transform(X_sample))
    return shap_values

# ============================
# 4. WHAT-IF 기반 서비스 추천
# ============================
def what_if_recommendations(model, customer_row, service_columns):
    """
    고객이 특정 서비스를 구독했을 때 이탈 확률 변화 계산
    """
    base = customer_row.copy()
    base_pred = model.predict_proba(base.to_frame().T)[0][1]

    rec_list = []

    for col in service_columns:
        temp = base.copy()

        # 서비스가 No/Off 인 경우만 변경
        if temp[col] in ["No", "False", 0]:
            temp[col] = "Yes"

            new_pred = model.predict_proba(temp.to_frame().T)[0][1]
            delta = new_pred - base_pred # 음수일수록 이탈 감소

            rec_list.append((col, delta))

    # 정렬: 가장 큰 이탈 방어 요인
    rec_list.sort(key=lambda x: x[1])

    return rec_list, base_pred

# ==========================
# 5. 특정 고객에게 추천 생성
# ==========================
def recommend_for_customer(model, df, customer_id):
    row = df[df["CustomerID"] == customer_id].iloc[0]

    service_cols = [
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
        "Phone Service",
        "Multiple Lines",
        "Paperless Billing",
        "Internet Service",
        "Contract",
        "Payment Method",
    ]

    recs, pred = what_if_recommendations(model, row, service_cols)

    print(f"\n고객 {customer_id}의 현재 이탈 확률: {pred:.3f}")
    print("\n=== 추천 서비스 (이탈 감소 효과 높은 순) ===")
    for service, delta in recs[:5]:
        print(f"{service}: {(delta):.3f}")

    return recs, pred


if __name__ == '__main__':
    # 1. 데이터 로드
    df = pd.read_csv("data/raw/telco2.csv")

    # 2. 모델 학습
    model, X, y, cat_cols, num_cols = train_churn_model(df)

    print(f"데이터 개수: {len(df)}")
    print(f"서비스 컬럼 수: {len(cat_cols)}")

    # 3. 특정 고객에게 서비스 추천 실행
    sample_customer = df["CustomerID"].iloc[0]   # 첫 번째 고객

    print("\n===============================")
    print(f"고객 {sample_customer} 추천 분석")
    print("===============================")

    recommend_for_customer(model, df, sample_customer)