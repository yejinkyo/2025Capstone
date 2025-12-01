import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import joblib
import os
from typing import List, Dict, Any, Tuple

# ======================================================================
# 1. 상수 정의 (NUMERICAL_COLS 및 ResourcesTuple 재정의)
# ======================================================================

MODEL_FEATURE_LIST = [
    'Gender', 'Age', 'Married', 'Dependents', 'noDependents', 
    'Referrals', 'noReferrals', 'PaperlessBilling', 
    'OnlineSecurity', 'OnlineBackup', 'TechSupport', 'UnlimitedData', 
    'AvgDownloadGB', 'CustomerLTV', 'SatisScore', 
    'TotalExtraDataCharge', 'AvgRoamCharge', 'TotalRoamCharge', 
    'Tenure_month', 'Sum_charge', 'Monthly_charge', 'ServiceDuration', 
    'CLTV_monthly', 'TotalOtherCharges', 'LTVPerSatis', 'Is_Manual_Payment',
    'PaymentMethod_신용카드', 'PaymentMethod_이체/메일확인',
    'AgeGroup_30대', 'AgeGroup_40대', 'AgeGroup_50대', 
    'AgeGroup_60대', 'AgeGroup_70대', 'AgeGroup_80대'
]

# 스케일링이 필요한 수치형 피처 목록 정의 (훈련 시 사용된 17개 피처)
NUMERICAL_COLS = [
    'Age', 'AvgDownloadGB', 'CustomerLTV', 'TotalExtraDataCharge',
    'AvgRoamCharge', 'TotalRoamCharge', 
    'noReferrals', 'noDependents', 'SatisScore', 'Tenure_month', 'Sum_charge', 'Monthly_charge', 'ServiceDuration',
    'CLTV_monthly', 'TotalOtherCharges', 'LTVPerSatis', 'Is_Manual_Payment'
]

VALUE_MAPPING = {
    '남자': '남성', '남': '남성', 'male': '남성', '여자': '여성', '여': '여성', 'female': '여성',
    '가입': 'Yes', '사용': 'Yes', '예': 'Yes', 'true': 'Yes', '1': 'Yes',
    '미가입': 'No', '미사용': 'No', '아니요': 'No', 'false': 'No', '0': 'No',
    '신용카드': '신용카드', '계좌이체': '계좌이체', '이체': '이체/메일확인', '메일확인': '이체/메일확인', '이체/메일확인': '이체/메일확인'
}

BINARY_COLS = ['Gender', 'Married', 'Dependents', 'Referrals', 'PaperlessBilling',
               'OnlineSecurity', 'OnlineBackup', 'TechSupport', 'UnlimitedData']

# ResourcesTuple: (lr_model, sbert, corpus_emb, df_text, df_cluster, scaler)
ResourcesTuple = Tuple[Any, SentenceTransformer, np.ndarray, pd.DataFrame, pd.DataFrame, Any]

# ======================================================================
# 2. 리소스 로드 함수 (Scaler 로직 추가)
# ======================================================================

def init_contrastive_resources() -> ResourcesTuple:
    """대조 분석에 필요한 모든 리소스를 로드합니다."""
    paths = {
        'lr': 'data/processed/lr_model.joblib',
        'scaler': 'data/processed/scaler.joblib', # Scaler 경로 추가
        'emb': 'data/processed/corpus_embeddings.joblib',
        'text': 'data/processed/telco_narrative_corpus.csv',
        'cluster': 'data/processed/telco_cleaned_data.csv'
    }
    
    print("📦 리소스 로딩 중...")
    try:
        lr_model = joblib.load(paths['lr'])
        # Scaler 로드 추가
        scaler = joblib.load(paths['scaler'])
        corpus_emb = joblib.load(paths['emb'])
        df_text = pd.read_csv(paths['text'])
        df_cluster = pd.read_csv(paths['cluster'])
        sbert = SentenceTransformer('jhgan/ko-sroberta-multitask')
        print("✅ 리소스 로드 완료.")
        # Scaler를 반환 튜플에 포함
        return lr_model, sbert, corpus_emb, df_text, df_cluster, scaler
    except Exception as e:
        print(f"❌ [오류] 리소스 로드 실패: {e}")
        # Scaler 포함하여 None 반환
        return None, None, None, None, None, None

# ======================================================================
# 3. 전처리 및 예측 유틸리티 함수 (Scaler 로직 추가)
# ======================================================================

def get_customer_features_by_id(user_id: str, df_cluster: pd.DataFrame) -> Dict[str, Any]:
    """군집 데이터프레임에서 특정 user_id의 피처를 추출합니다."""
    id_col_cluster = None
    for col in ['CustomerID', 'customerID', 'id', 'CustomerId']:
        if col in df_cluster.columns: id_col_cluster = col; break
    if not id_col_cluster: return {}
    
    row = df_cluster[df_cluster[id_col_cluster].astype(str) == str(user_id)]
    if row.empty: return {} 
        
    non_feature_cols = ['Churn Label', 'ChurnLabel', 'kmeans_cluster_id', id_col_cluster]
    features = row.iloc[0].to_dict()
    return {k: v for k, v in features.items() if k not in non_feature_cols}


def process_user_input_to_df(A_features_raw: Dict[str, Any]) -> pd.DataFrame:
    """사용자 입력을 받아 모델 학습 데이터와 동일한 형태의 DataFrame을 만듭니다."""
    feature_data = {col: 0 for col in MODEL_FEATURE_LIST}
    for raw_key, raw_value in A_features_raw.items():
        clean_val = str(raw_value).strip()
        
        if raw_key == 'Gender': feature_data['Gender'] = 1 if VALUE_MAPPING.get(clean_val, clean_val) in ['남성', 'Male'] else 0
        elif raw_key in BINARY_COLS: feature_data[raw_key] = 1 if VALUE_MAPPING.get(clean_val.lower(), clean_val).lower() in ['yes', '1'] else 0
        elif raw_key == 'Dependents': 
            is_yes = 1 if VALUE_MAPPING.get(clean_val.lower(), clean_val).lower() in ['yes', '1'] else 0
            feature_data['Dependents'] = is_yes; feature_data['noDependents'] = 1 - is_yes
        elif raw_key == 'Referrals': 
            is_yes = 1 if VALUE_MAPPING.get(clean_val.lower(), clean_val).lower() in ['yes', '1'] else 0
            feature_data['Referrals'] = is_yes; feature_data['noReferrals'] = 1 - is_yes
        elif raw_key in feature_data:
            try: feature_data[raw_key] = float(clean_val)
            except: pass

        if raw_key == 'Age':
            try:
                age = int(float(clean_val)); feature_data['Age'] = age
                decade = (age // 10) * 10
                if decade >= 30: 
                    target_col = f"AgeGroup_{decade}대"
                    if target_col in feature_data: feature_data[target_col] = 1
            except: pass
        if raw_key == 'PaymentMethod':
            target_col = f"PaymentMethod_{VALUE_MAPPING.get(clean_val, clean_val)}"
            if target_col in feature_data: feature_data[target_col] = 1

    # 파생변수 자동 계산
    feature_data['TotalOtherCharges'] = feature_data.get('TotalExtraDataCharge', 0) + feature_data.get('TotalRoamCharge', 0)
    ltv = feature_data.get('CustomerLTV', 0); dur = feature_data.get('ServiceDuration', 1); satis = feature_data.get('SatisScore', 1)
    if dur == 0: dur = 1; 
    if satis == 0: satis = 1;
    feature_data['CLTV_monthly'] = ltv / dur
    feature_data['LTVPerSatis'] = ltv / satis
    is_manual = 0
    if feature_data.get('PaymentMethod_신용카드') == 1 or feature_data.get('PaymentMethod_이체/메일확인') == 1: is_manual = 1
    feature_data['Is_Manual_Payment'] = is_manual

    return pd.DataFrame([feature_data], columns=MODEL_FEATURE_LIST)


def predict_churn_probability(model: Any, A_input_df: pd.DataFrame, scaler: Any) -> float:
    """모델을 사용해 이탈 확률을 예측합니다. (Scaler 적용)"""
    
    A_input_scaled_df = A_input_df.copy()

    try:
        # 1. 수치형 피처만 추출하여 스케일링 적용
        cols_to_scale = [col for col in NUMERICAL_COLS if col in A_input_scaled_df.columns]
        
        if scaler is not None and cols_to_scale:
            numerical_data = A_input_scaled_df[cols_to_scale]
            # 학습 시 사용한 Scaler의 transform만 적용
            scaled_data = scaler.transform(numerical_data)
            
            # 2. 스케일링된 데이터로 교체
            A_input_scaled_df[cols_to_scale] = scaled_data

        # 3. 모델 예측
        churn_prob = model.predict_proba(A_input_scaled_df)[:, 1][0]
        return float(churn_prob)
    except Exception as e:
        print(f"[경고] 모델 예측 중 오류 발생 (스케일링 포함): {e}. 기본값 0.5 반환.")
        return 0.5


def find_retained_neighbors(b_id: Any, df_cluster: pd.DataFrame) -> pd.DataFrame:
    """B와 같은 군집에 있는 비이탈 고객들을 찾습니다."""
    id_col = 'CustomerID'
    for col in df_cluster.columns:
        if col.lower() in ['customerid', 'id']: id_col = col; break
    
    try:
        b_row = df_cluster[df_cluster[id_col].astype(str) == str(b_id)]
        if b_row.empty: return pd.DataFrame()
        b_cluster = b_row['kmeans_cluster_id'].values[0]
        cluster_id = b_cluster
    except:
        return pd.DataFrame()

    churn_col = 'Churn Label' if 'Churn Label' in df_cluster.columns else 'ChurnLabel'
    neighbors = df_cluster[
        (df_cluster['kmeans_cluster_id'] == cluster_id) & 
        (df_cluster.get(churn_col, 'No') == 'No')
    ]
    return neighbors


def find_most_similar_customer_B(A_consult_text: str, model: SentenceTransformer, corpus_embeddings: np.ndarray, df_telco_text_raw: pd.DataFrame) -> Tuple[Any, float]:
    """A의 상담 텍스트와 가장 유사한 B(1명)를 S-BERT로 찾습니다."""
    A_embedding = model.encode([A_consult_text], normalize_embeddings=True)
    sim_scores = util.cos_sim(A_embedding, corpus_embeddings)[0].numpy()
    b_index = np.argmax(sim_scores)
    b_customer_row = df_telco_text_raw.iloc[b_index]
    b_id = None
    for name in ['CustomerID', 'customerID', 'id', 'CustomerId']:
        if name in b_customer_row.index: b_id = b_customer_row[name]; break
    if b_id is None: b_id = b_customer_row.iloc[0]
    return b_id, float(sim_scores[b_index])

# ======================================================================
# 4. 대조 분석 실행 함수 (Scaler 로직 추가 및 결과 형식 수정)
# ======================================================================

def perform_contrastive_analysis_for_user(user_id, consult_text, resources):
    # Scaler를 포함한 6개 요소 언팩
    lr_model, sbert, corpus_emb, df_text, df_cluster, scaler = resources
    
    churn_prob = 0.0
    
    fail_response = {"role_model_pattern": "분석 불가 (데이터 부족)", "insight": "데이터 부족"}
    if sbert is None: return fail_response

    try:
        raw_features = get_customer_features_by_id(user_id, df_cluster)
        if not raw_features:
             return {"role_model_pattern": "분석 불가 (ID 오류)", "insight": "고객 ID 정보가 데이터에 없습니다."}
             
        A_df = process_user_input_to_df(raw_features)
        
        # **<--- 핵심 로직: 이탈 예측 및 조건 확인 (Scaler 전달하여 호출) --->**
        churn_prob = predict_churn_probability(lr_model, A_df, scaler) # Scaler 전달
        
        if churn_prob <= 0.5:
            print(f"[알림] 예측 이탈 확률 ({churn_prob:.2%})이 50% 이하이므로 대조 분석을 생략합니다.")
            return {
                "role_model_pattern": "저위험군",
                "insight": "이탈 확률이 기준(50%) 이하입니다."
            }
        # **<--- 핵심 로직 끝 --->**


        # 1. 유사 고객(Role Model) 찾기 
        if not consult_text: consult_text = "서비스 불만 및 해지 고민"

        target_emb = sbert.encode([consult_text], normalize_embeddings=True)
        sim_scores = util.cos_sim(target_emb, corpus_emb)[0].numpy()
        best_idx = np.argmax(sim_scores)
        b_row = df_text.iloc[best_idx]
        
        b_id = None
        for col_name in ['CustomerID', 'customerID', 'id', 'CustomerId']:
            if col_name in b_row.index: b_id = b_row[col_name]; break
        if b_id is None:
            return {"role_model_pattern": "유사 사례 매칭 실패", "insight": "유사 텍스트를 찾았으나 ID 매칭 실패"}

        # 2. 유사 군집 및 롤모델 찾기 
        id_col_cluster = None
        for col in ['CustomerID', 'customerID', 'id', 'CustomerId']:
            if col in df_cluster.columns: id_col_cluster = col; break
        
        neighbors = find_retained_neighbors(b_id, df_cluster)

        # 3. 🚩 차이점 분석 (수정된 추천 로직 유지)
        service_recommendations = []
        payment_recommendation = ""
        target_services = ['OnlineSecurity', 'OnlineBackup', 'TechSupport', 'UnlimitedData', 'StreamingTV']
        
        my_row = df_cluster[df_cluster[id_col_cluster].astype(str) == str(user_id)]
        
        if not my_row.empty:
            # 3-1. 서비스 가입 추천 (나는 미가입 AND 이웃 중 최소 1명 이상 가입 시 추천)
            for col in target_services:
                if col in my_row.columns and col in neighbors.columns:
                    my_val = my_row[col].values[0]
                    
                    # 1. 나는 미가입 상태 (No/0/false/nan)
                    if str(my_val).lower() in ['no', '0', 'false', 'nan']:
                        # 2. 비이탈 유사 그룹의 평균 사용률 계산
                        group_usage = neighbors[col].apply(lambda x: 1 if str(x).lower() in ['yes', '1', 'true'] else 0).mean()
                        
                        # 수정된 기준: 사용률이 0% 초과일 때 추천
                        if group_usage > 0.0:
                            service_recommendations.append(f"{col}")
            
            # 3-2. 결제 수단 변경 추천 (유사 그룹 최빈값 기준)
            if 'PaymentMethod' in my_row.columns and 'PaymentMethod' in neighbors.columns:
                my_payment = my_row['PaymentMethod'].values[0]
                neighbor_mode = neighbors['PaymentMethod'].mode()
                if not neighbor_mode.empty:
                    neighbor_preferred_payment = neighbor_mode[0]
                    if str(my_payment) != str(neighbor_preferred_payment):
                        payment_recommendation = f"{neighbor_preferred_payment}"

        # 4. 결과 출력 문자열 조합
        recommendation_parts = []
        if service_recommendations:
            recommendation_parts.extend(service_recommendations)
        if payment_recommendation:
            recommendation_parts.append(payment_recommendation)
        
        recommendations_str = ", ".join(recommendation_parts) 

        if not recommendations_str:
            return {
                "role_model_pattern": "기본 요금제 유지",
                "insight": "유사 고객들은 현재 상태에 만족하고 있습니다."
            }
        
        return {
            "role_model_pattern": recommendations_str,
            "insight": f"유사한 만족 고객들은 {recommendations_str} 등을 이용중입니다."
        }

    except Exception as e:
        print(f"[대조분석] 로직 실행 중 에러: {e}")
        return {"role_model_pattern": "분석 중 기술적 오류", "insight": str(e)}
    

# ======================================================================
# 5. 테스트 실행 블록 (무작위 30명 테스트 및 결과 출력)
# ======================================================================
if __name__ == "__main__":
    print("--- 대조 분석 모듈 테스트 시작 ---")

    contrast_resources = init_contrastive_resources()
    
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
    user_consult_text = "요금이 좀 비싼 것 같고, 폰이 자주 고장나서 걱정이에요."
    user_df_contrast = pd.DataFrame([target_user])

    contrast_result = perform_contrastive_analysis_for_user(
    user_id="C-10008",
    consult_text=user_consult_text,
    resources=contrast_resources
)
    
    print(contrast_result)