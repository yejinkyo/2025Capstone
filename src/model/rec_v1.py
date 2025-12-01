import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import joblib
import os
from typing import List, Dict, Any, Tuple

# ======================================================================
# 1. 상수 정의
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

# Scaler가 fit된 정확한 17개 컬럼 목록 (transform 대상)
SCALE_COLS_FOR_TRANSFORM = [
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

BINARY_COLS = ['Married', 'PaperlessBilling', 'OnlineSecurity', 
               'OnlineBackup', 'TechSupport', 'UnlimitedData']

# ResourcesTuple: (lr_model, sbert, corpus_emb, df_text, df_cluster, scaler) - 6개 요소
ResourcesTuple = Tuple[Any, SentenceTransformer, np.ndarray, pd.DataFrame, pd.DataFrame, Any]

# ======================================================================
# 2. 리소스 로드 함수 (6개 요소 반환)
# ======================================================================

def init_contrastive_resources(paths: Dict[str, str]) -> ResourcesTuple:
    """모든 리소스와 Scaler를 로드합니다."""
    print("📦 리소스 로딩 중...")
    try:
        lr_model = joblib.load(paths['lr'])
        corpus_emb = joblib.load(paths['emb'])
        df_text = pd.read_csv(paths['text'])
        df_cluster = pd.read_csv(paths['cluster'])
        sbert = SentenceTransformer('jhgan/ko-sroberta-multitask')
        
        # 🚩 Scaler 로드 시도
        scaler = None
        if 'scaler' in paths and os.path.exists(paths['scaler']):
            scaler = joblib.load(paths['scaler'])
            print("✅ Scaler 객체 로드 완료.")
        else:
            # Scaler 파일이 없으면 오류 메시지 출력
            print("❌ [경고] Scaler 파일이 없어 정확한 예측 불가능. Raw Data로 예측 시도합니다.")
            
        print("✅ 리소스 로드 완료.")
        # 6개 요소 반환
        return lr_model, sbert, corpus_emb, df_text, df_cluster, scaler
    except Exception as e:
        print(f"❌ [오류] 리소스 로드 실패: {e}")
        return None, None, None, None, None, None

def load_resources():
    """테스트 블록의 요구사항에 맞춰 init_contrastive_resources를 경로 없이 호출"""
    RESOURCE_PATHS = {
        'lr': 'data/processed/lr_model.joblib',
        'emb': 'data/processed/corpus_embeddings.joblib',
        'text': 'data/processed/telco_narrative_corpus.csv',
        'cluster': 'data/processed/telco_cleaned_data.csv',
        'scaler': 'data/processed/scaler.joblib' # Scaler 경로 추가
    }
    return init_contrastive_resources(RESOURCE_PATHS)

# ======================================================================
# 3. 전처리 및 예측 유틸리티 함수
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
    if model is None: return 0.0
    
    A_input_scaled_df = A_input_df.copy()

    try:
        # Scaler가 있을 경우에만 변환 수행
        if scaler is not None:
            cols_to_scale = SCALE_COLS_FOR_TRANSFORM
            numerical_data = A_input_scaled_df[cols_to_scale]
            scaled_data = scaler.transform(numerical_data)
            A_input_scaled_df[cols_to_scale] = scaled_data
            
        # 예측
        churn_prob = model.predict_proba(A_input_scaled_df)[:, 1][0]
        return float(churn_prob)
            
    except Exception as e:
        print(f"[경고] 모델 예측 중 오류 발생 (스케일링 문제): {e}. 0.0 반환.")
        return 0.0 # 예측 실패 시 0.0% 반환


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
# 4. 대조 분석 실행 함수 (New User 피처 주입 로직 포함)
# ======================================================================

def perform_contrastive_analysis_for_user(user_id, consult_text, resources, scaler, user_features_dict=None):
    # 5개 요소 언팩
    lr_model, sbert, corpus_emb, df_text, df_cluster = resources
    
    fail_response = {"role_model_pattern": "분석 불가 (데이터 부족)", "insight": "데이터 부족", "churn_probability": 0.0}
    if sbert is None: return fail_response

    try:
        # 1. 피처 데이터 준비: user_features_dict가 있으면 NewUser로 간주하고 피처를 사용
        if user_features_dict:
            raw_features = user_features_dict
        else:
            # 기존 고객의 경우 ID 기반 조회 시도
            raw_features = get_customer_features_by_id(user_id, df_cluster)
            
        if not raw_features:
             return {"role_model_pattern": "분석 불가 (ID 오류)", "insight": "고객 ID 정보가 데이터에 없습니다.", "churn_probability": 0.0}
             
        A_df = process_user_input_to_df(raw_features)
        
        # 2. 이탈 예측 (Scaler 전달)
        churn_prob = predict_churn_probability(lr_model, A_df, scaler) 
        
        if churn_prob <= 0.5:
            print(f"[알림] 예측 이탈 확률 ({churn_prob:.2%})이 50% 이하이므로 대조 분석을 생략합니다.")
            return {
                "role_model_pattern": "저위험군",
                "insight": "이탈 확률이 기준(50%) 이하입니다.",
                "churn_probability": float(f"{churn_prob:.4f}")
            }
        
        # 3. 유사 고객(Role Model) 찾기 
        if not consult_text: consult_text = "서비스 불만 및 해지 고민"

        target_emb = sbert.encode([consult_text], normalize_embeddings=True)
        sim_scores = util.cos_sim(target_emb, corpus_emb)[0].numpy()
        best_idx = np.argmax(sim_scores)
        b_row = df_text.iloc[best_idx]
        
        b_id = None
        for col_name in ['CustomerID', 'customerID', 'id', 'CustomerId']:
            if col_name in b_row.index: b_id = b_row[col_name]; break
        if b_id is None:
            return {"role_model_pattern": "유사 사례 매칭 실패", "insight": "유사 텍스트를 찾았으나 ID 매칭 실패", "churn_probability": float(f"{churn_prob:.4f}")}

        # 4. 유사 군집 및 롤모델 찾기 
        id_col_cluster = None
        for col in ['CustomerID', 'customerID', 'id', 'CustomerId']:
            if col in df_cluster.columns: id_col_cluster = col; break
        
        neighbors = find_retained_neighbors(b_id, df_cluster)

        # 5. 🚩 차이점 분석 (개선된 추천 로직)
        service_recommendations = []
        payment_recommendation = ""
        target_services = ['OnlineSecurity', 'OnlineBackup', 'TechSupport', 'UnlimitedData', 'PaperlessBilling']
        
        
        if raw_features:
            # 5-1. 서비스 가입 추천 (유사 그룹 50% 이상 사용, 나는 미가입)
            for col in target_services:
                my_val = raw_features.get(col, '미가입')
                
                if col in neighbors.columns:
                    if str(my_val).lower() in ['no', '미가입', '0', 'false', 'nan']:
                        group_usage = neighbors[col].apply(lambda x: 1 if str(x).lower() in ['yes', '1', 'true'] else 0).mean()
                        if group_usage >= 0.5: # 🚩 50% 기준 적용
                            service_recommendations.append(f"서비스: {col}")
            
            # 5-2. 결제 수단 변경 추천 (유사 그룹 최빈값 기준)
            my_payment = raw_features.get('PaymentMethod', '계좌이체') 
            if 'PaymentMethod' in neighbors.columns:
                neighbor_mode = neighbors['PaymentMethod'].mode()
                if not neighbor_mode.empty:
                    neighbor_preferred_payment = neighbor_mode[0]
                    if str(my_payment) != str(neighbor_preferred_payment):
                        payment_recommendation = f"결제수단: '{neighbor_preferred_payment}'"

        # 6. 결과 출력 문자열 조합
        recommendation_parts = []
        if service_recommendations:
            recommendation_parts.extend(service_recommendations)
        if payment_recommendation:
            recommendation_parts.append(payment_recommendation)
        
        recommendations_str = ", ".join(recommendation_parts) 

        if not recommendations_str:
            return {
                "role_model_pattern": "기존 요금제 사용",
                "insight": "유사한 만족 고객들은 현재 고객(A)과 피처 차이가 거의 없습니다.",
                "churn_probability": float(f"{churn_prob:.4f}")
            }
        
        return {
            "role_model_pattern": recommendations_str,
            "insight": f"유사한 만족 고객들은 {recommendations_str} 등을 이용중입니다.",
            "churn_probability": float(f"{churn_prob:.4f}")
        }

    except Exception as e:
        print(f"[대조분석] 로직 실행 중 에러: {e}")
        return {"role_model_pattern": "분석 중 기술적 오류", "insight": str(e), "churn_probability": 0.0}
    

def generate_recommendations_contrast(user_text, user_feats, lr_model, sbert_model, corpus_embeddings, df_text, df_cluster):
    """
    NewUser 테스트를 위한 래퍼 함수: perform_contrastive_analysis_for_user의 시그니처에 맞게 데이터를 재구성하여 호출합니다.
    """
    # 5개 리소스를 perform_contrastive_analysis_for_user에 전달
    resources = (lr_model, sbert_model, corpus_embeddings, df_text, df_cluster)
    
    # NewUser의 CustomerID를 추출 (ID는 분석 로직 내에서 필요)
    user_id = user_feats.get('CustomerID', 'NewUser')

    # Scaler 로드 시도 (Scaler는 리소스로드 시점에서는 로드되지 않았으므로 여기서 로드)
    scaler_path = 'data/processed/scaler.joblib'
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    # perform_contrastive_analysis_for_user 함수에 user_features_input 딕셔너리와 Scaler를 전달
    result = perform_contrastive_analysis_for_user(user_id, user_text, resources, scaler=scaler, user_features_dict=user_feats)
    
    # 출력은 generate_recommendations_contrast 내부에서 상세하게 출력
    print(f"--- 1. 입력 처리 완료: {user_text}")
    print(f"--- 2. 이탈 예측 확률: {result.get('churn_probability', 0.0) * 100:.2f}%")
    print("\n3. 분석 결과:")
    print(f"  - Role Model Pattern: {result.get('role_model_pattern', '오류')}")
    print(f"  - Insight: {result.get('insight', '분석 중 오류 발생')}")
    
    return result.get('churn_probability', 0.0)


if __name__ == "__main__":
    
    # NOTE: df 로드는 리소스 로드 함수 내에서 수행되어야 하므로 제거했습니다.
    
    try:
        # 1. 리소스 로드 (한 번만 수행)
        # ⚠️ load_resources가 6개 요소를 반환하도록 수정되었으므로, 변수 목록을 6개로 늘립니다.
        lr_model, sbert_model, corpus_embeddings, df_text, df_cluster, scaler_model = load_resources()

        if lr_model is not None:
            # 2. 사용자 입력 시나리오 (새로운 고객 피처 정의)
            user_text_input = "요금제가 너무 비싸서 부담스러워요."
            user_features_input = {
                'CustomerID': 'NewUser_Example', # 임의의 ID
                'Gender': '남자',              
                'Age': 30,                     
                'Married': 'No',
                'Dependents': 'No',
                'Referrals': 'No',
                'PaperlessBilling': 'Yes',
                'OnlineSecurity': '미가입',    # 추천 대상 서비스
                'OnlineBackup': '미가입',
                'TechSupport': '미가입',
                'UnlimitedData': 'No',
                'StreamingTV': '미가입',
                'PaymentMethod': '신용카드',   
                'AvgDownloadGB': 10.5,
                'CustomerLTV': 5000,
                'SatisScore': 3,
                'TotalExtraDataCharge': 50,
                'AvgRoamCharge': 10,
                'TotalRoamCharge': 100,
                'Tenure_month': 15,
                'Sum_charge': 1500,
                'Monthly_charge': 100,
                'ServiceDuration': 10
            }

            print(f"--- 대조 분석 모듈 테스트 시작 (New User) ---")
            print(f"상담 내용: {user_text_input}")
            
            # 3. 실행 (함수 호출)
            result_prob = generate_recommendations_contrast(
                user_text_input,
                user_features_input,
                lr_model,
                sbert_model,
                corpus_embeddings,
                df_text,
                df_cluster
            )
            
            # 최종 결과는 generate_recommendations_contrast 내부에서 출력됨

    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
