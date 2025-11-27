import pandas as pd
import shap
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

# =====================
# 1. 모델 학습 및 준비 (초기화)
# =====================
def init_shap_model(df_path):
    df = pd.read_csv(df_path)

    # 전처리
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    df = df.dropna(subset=["Total Charges"])
    
    # Feature 정의
    service_features = [
        "Online Security", "Online Backup", "Device Protection", "Tech Support",
        "Streaming TV", "Streaming Movies", "Phone Service", "Multiple Lines",
        "Paperless Billing", "Internet Service", "Contract", "Payment Method"
    ]
    numeric_features = ["Monthly Charges", "Total Charges", "Tenure Months"]
    categorical_features = service_features
    
    # 컬럼 존재 여부 확인 (에러 방지)
    existing_cat = [c for c in categorical_features if c in df.columns]
    existing_num = [c for c in numeric_features if c in df.columns]
    
    if not existing_cat or not existing_num:
        print("[SHAP] 필수 칼럼 부족")
        return None, df

    X = df[existing_cat + existing_num]
    y = df["Churn Value"]
    
    # 파이프라인 구축
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), existing_cat),
            ("num", "passthrough", existing_num),
        ]
    )
    
    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(max_iter=300)),
    ])
    
    model.fit(X, y)
    
    return model, df

# ============================
# 2. 특정 고객 분석 실행 함수
# ============================
def analyze_customer_shap(model, df, customer_id):
    """
    특정 고객의 이탈 확률과 개선 추천 사항을 반환
    """
    if model is None or df is None:
        return None

    # [수정됨] ID 검색 로직 강화 (타입 불일치 해결)
    # 1. 컬럼명 찾기
    id_col = 'CustomerID'
    if 'CustomerID' not in df.columns:
        if 'customerID' in df.columns: id_col = 'customerID'
        elif 'id' in df.columns: id_col = 'id'
    
    # 2. 강제 형변환 후 검색
    try:
        # 데이터프레임의 ID와 입력 ID 모두 문자열로 변환해서 공백 제거 후 비교
        mask = df[id_col].astype(str).str.strip() == str(customer_id).strip()
        if not mask.any():
            print(f"[SHAP] ID 매칭 실패: '{customer_id}' (데이터 내 유사 ID 없음)")
            return None
            
        row = df[mask].iloc[0]
    except Exception as e:
        print(f"[SHAP] 데이터 검색 중 오류: {e}")
        return None

    # 1. 현재 이탈 확률 계산
    # 모델 학습 때 사용한 컬럼만 골라서 입력
    features = model.named_steps['preprocess'].transformers_[0][2] + \
               model.named_steps['preprocess'].transformers_[1][2]
    
    input_data = row[features].to_frame().T
    base_pred = model.predict_proba(input_data)[0][1]

    # 2. What-If 분석
    service_cols = [
        "Online Security", "Online Backup", "Device Protection", "Tech Support",
        "Streaming TV", "Streaming Movies", "Paperless Billing", "Contract"
    ]
    
    rec_list = []
    for col in service_cols:
        if col not in df.columns: continue

        # 현재 미사용(No)인 서비스만 타겟
        val = str(row[col]).lower()
        if val in ["no", "false", "0", "month-to-month"]:
            temp = row.copy()
            # 서비스별 긍정 값 설정
            if col == "Contract": temp[col] = "One year"
            else: temp[col] = "Yes" 
            
            # 예측
            temp_input = temp[features].to_frame().T
            new_pred = model.predict_proba(temp_input)[0][1]
            delta = new_pred - base_pred 
            
            if delta < 0:
                rec_list.append((col, delta))

    rec_list.sort(key=lambda x: x[1])

    if base_pred > 0.7:
        pain_point = "높은 이탈 위험도"
    else:
        pain_point=""

    top_recs = [item[0] for item in rec_list[:3]]
    
    return {
        "churn_prob": base_pred,
        "pain_point": pain_point,
        "detail": f"현재 이탈 위험 {base_pred:.1%}. {', '.join(top_recs)} 가입 시 위험 감소 예상.",
        "top_recommendations": top_recs
    }