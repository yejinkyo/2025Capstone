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

# ==============================
# 0. 데이터 변환
# ==============================
def adapt_telco_to_shap(contrast_row_dict):
    """
    대조분석용 데이터를 SHAP 모델용 포맷으로 변환
    """
    # 값 매핑
    val_map = {
        'Yes': 'Yes', 'No': 'No', 'True': 'Yes', 'False': 'No',
        '가입': 'Yes', '미가입': 'No', '사용': 'Yes', '미사용': 'No',
        '남자': 'Male', '여자': 'Female', '남성': 'Male', '여성': 'Female',
        '신용카드': 'Credit card (automatic)',
        '계좌이체': 'Bank transfer (automatic)',
        '이체/메일확인': 'Mailed check', 
        '메일확인': 'Mailed check',
        '전자수표': 'Electronic check',
        'None': 'No' 
    }

    shap_data = {}
    
    # --- 1. 범주형 데이터 매핑 ---
    # (1) 기본 정보
    shap_data['Gender'] = val_map.get(str(contrast_row_dict.get('Gender')), 'Male')
    shap_data['Partner'] = val_map.get(str(contrast_row_dict.get('Married')), 'No') 
    shap_data['Dependents'] = val_map.get(str(contrast_row_dict.get('Dependents')), 'No')
    shap_data['Senior Citizen'] = 0 
    
    # (2) 서비스 정보 (여기가 중요!)
    shap_data['Online Security'] = val_map.get(str(contrast_row_dict.get('OnlineSecurity', 'No')), 'No')
    shap_data['Online Backup'] = val_map.get(str(contrast_row_dict.get('OnlineBackup', 'No')), 'No')
    shap_data['Tech Support'] = val_map.get(str(contrast_row_dict.get('TechSupport', 'No')), 'No')
    shap_data['Paperless Billing'] = val_map.get(str(contrast_row_dict.get('PaperlessBilling', 'Yes')), 'Yes')
    shap_data['Payment Method'] = val_map.get(str(contrast_row_dict.get('PaymentMethod')), 'Electronic check')
    shap_data['Device Protection'] = val_map.get(str(contrast_row_dict.get('Device Protection', 'No')), 'No')
    shap_data['Streaming TV'] = val_map.get(str(contrast_row_dict.get('Streaming TV', 'No')), 'No')
    shap_data['Streaming Movies'] = val_map.get(str(contrast_row_dict.get('Streaming Movies', 'No')), 'No')
    
    shap_data['Contract'] = val_map.get(str(contrast_row_dict.get('Contract', 'Month-to-month')), 'Month-to-month')
    shap_data['Phone Service'] = 'Yes' 
    shap_data['Multiple Lines'] = 'No'
    
    if str(contrast_row_dict.get('UnlimitedData')) in ['Yes']:
        shap_data['Internet Service'] = 'Fiber optic'
    else:
        shap_data['Internet Service'] = 'DSL'

    # --- 2. 수치형 데이터 변환 (원화 -> 달러 환율 적용) ---
    shap_data['Monthly Charges'] = float(contrast_row_dict.get('Monthly_charge', 0))
    shap_data['Total Charges'] = float(contrast_row_dict.get('Sum_charge', 0))
    shap_data['Tenure Months'] = int(contrast_row_dict.get('Tenure_month', 0))

    return pd.DataFrame([shap_data])

# ============================
# 2. 특정 고객 분석 실행 함수
# ============================
def analyze_customer_shap(model, df, user_id=None, custom_data=None):
    """
    custom_data: adapt_telco_to_shap 함수를 통해 변환된 1줄짜리 DataFrame
    """
    row = None
    
    # 외부에서 변환된 데이터
    if custom_data is not None:
        try:
            features = model.named_steps['preprocess'].transformers_[0][2] + \
                       model.named_steps['preprocess'].transformers_[1][2]
            
            for f in features:
                if f not in custom_data.columns:
                    custom_data[f] = 0 if f in ["Monthly Charges", "Total Charges", "Tenure Months"] else "No"
            
            row = custom_data.iloc[0]
        except Exception as e:
            print(f"[SHAP] 커스텀 데이터 오류: {e}")
            return None

    # ID 검색
    elif user_id is not None:
        id_col = 'CustomerID'
        try:
            mask = df[id_col].astype(str).str.strip() == str(user_id).strip()
            if not mask.any(): return None
            row = df[mask].iloc[0]
        except: return None
    else:
        return None

    # 1. 현재 이탈 확률 계산
    features = model.named_steps['preprocess'].transformers_[0][2] + \
               model.named_steps['preprocess'].transformers_[1][2]
    
    input_data = row[features].to_frame().T
    base_pred = model.predict_proba(input_data)[0][1]

    print(f"DEBUG: 기본 이탈 확률 = {base_pred:.4f}") # 디버깅용 출력

    # 2. What-If 분석 (추천 로직)
    target_services = [
        "Contract",      
        "Tech Support",
        "Online Security", 
        "Online Backup", 
        "Device Protection",
        "Paperless Billing" 
    ]
    
    rec_list = []
    
    for col in target_services:
        val = str(row.get(col, 'No')).strip().lower() # 공백제거 및 소문자
        
        if val in ["no", "false", "0", "month-to-month"]:
            temp = row.copy()
            
            if col == "Contract": temp[col] = "One year"
            elif col == "Paperless Billing": temp[col] = "No" 
            else: temp[col] = "Yes" 
            
            # 예측
            temp_input = temp[features].to_frame().T
            new_pred = model.predict_proba(temp_input)[0][1]
            delta = new_pred - base_pred 
            
            print(f"DEBUG: {col} 변경 시 변화량 = {delta:.5f}") # 디버깅
            
            # [디버깅]
            if delta < 0:
                rec_list.append((col, delta))
            elif base_pred >= 0.99 and delta <= 0:
                rec_list.append((col, -0.001)) 

    # 3. 결과 정리
    if not rec_list:
        top_recs = []
        detail_msg = "이탈 위험이 매우 높습니다. 장기 약정 및 기술 지원 서비스 제안이 시급합니다."
    else:
        rec_list.sort(key=lambda x: x[1]) 
        top_recs = [item[0] for item in rec_list[:3]]
        
        # 한글화 매핑 
        name_map = {"Contract": "약정 연장", "Tech Support": "기술 지원", "Online Security": "온라인 보안", "Online Backup": "온라인 백업"}
        display_recs = [name_map.get(r, r) for r in top_recs]
        
        detail_msg = f"현재 이탈 위험 {base_pred:.1%}. {', '.join(top_recs)} 제안 시 위험 감소 예상."

    if base_pred > 0.7:
        pain_point = "높은 이탈 위험도"
    else:
        pain_point = "안정적"

    return {
        "churn_prob": base_pred,
        "pain_point": pain_point,
        "detail": detail_msg,
        "top_recommendations": top_recs
    }


if __name__ == '__main__':
    user_high_risk = {
        'CustomerId': 'Test-001',
        'Gender': '남성',               # telco.csv 스타일
        'Age': 35,
        'Married': 'No',               # Partner로 변환될 예정
        'Dependents': 'No',
        'Tenure_month': 2,             # 가입 2개월차 (이탈 위험 높음)
        'Monthly_charge': 95000,       # 고가 요금제
        'Sum_charge': 190000,
        'OnlineSecurity': 'Yes',        # 미가입 -> 추천 대상
        'OnlineBackup': 'Yes',
        'TechSupport': 'Yes',           # 미가입 -> 추천 대상
        'UnlimitedData': 'Yes',        # 인터넷 사용 중 (Fiber optic 추정)
        'PaperlessBilling': 'Yes',
        'PaymentMethod': '신용카드' ,
        'Device Protection': 'Yes'
    }

    # 2. SHAP용으로 데이터 변환 (새로 만든 함수)
    model_shap, df_shap_origin = init_shap_model('data/raw/telco2.csv') # 모델 로드
    shap_input_df = adapt_telco_to_shap(user_high_risk) # 변환!

    # 3. 분석 실행
    shap_result = analyze_customer_shap(model_shap, df_shap_origin, custom_data=shap_input_df)

    print(shap_result)

    user_loyal = {
    'CustomerId': 'Test-002',
    'Gender': '여성',
    'Age': 83,
    'Married': 'Yes',
    'Dependents': 'Yes',
    'Tenure_month': 50,    
    'Monthly_charge': 59,
    'Sum_charge': 300,
    'OnlineSecurity': 'Yes',      
    'OnlineBackup': 'Yes',
    'TechSupport': 'Yes',
    'UnlimitedData': 'Yes',
    'PaperlessBilling': 'No',
    'PaymentMethod': '계좌이체',
    }   

    target_user = {
    'CustomerId': 'Demo-User-001',
    'Gender': '여성',
    'Age': 34,
    'Married': 'No',
    'Dependents': 'No',
    'Tenure_month': 12,           # 1년차 고객
    'Monthly_charge': 75000,      # 비교적 높은 요금
    'Sum_charge': 900000,
    'OnlineSecurity': 'No',       # [기회] 보안 서비스 없음
    'OnlineBackup': 'No',
    'TechSupport': 'No',          # [기회] 기술 지원 없음
    'UnlimitedData': 'Yes',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': '신용카드',
    'Device Protection': 'No'     # [기회] 기기 보호 없음
    }

    shap_input_df = adapt_telco_to_shap(target_user)

    # 3. 분석 실행
    shap_result = analyze_customer_shap(model_shap, df_shap_origin, custom_data=shap_input_df)

    print(shap_result)