import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import joblib
import os
import warnings
from typing import List, Dict, Any, Tuple

warnings.filterwarnings('ignore')

# ======================================================================
# 1. 상수 정의 및 경로 지정
# ======================================================================

WEIGHT_STRUCT = 0.7
WEIGHT_TEXT = 0.3
LR_MODEL_PATH = 'data/processed/lr_model.joblib' # 정형 모델 파일
SENTIMENT_MODEL_PATH = 'data/processed/sentiment_model.joblib' # 감성 모델 파일

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

RESOURCE_PATHS = {
    'lr': LR_MODEL_PATH,
    'sentiment': SENTIMENT_MODEL_PATH, 
    'emb': 'data/processed/corpus_embeddings.joblib',
    'text': 'data/processed/telco_narrative_corpus.csv',
    'cluster': 'data/processed/telco_cleaned_data.csv',
    'scaler': 'data/processed/scaler.joblib' 
}


# ======================================================================
# 2. 하이브리드 예측기 클래스 정의
# ======================================================================

class HybridChurnPredictor:
    """정형 데이터와 텍스트 감성 분석을 결합한 이탈 예측기"""
    def __init__(self, lr_path=LR_MODEL_PATH, sentiment_path=SENTIMENT_MODEL_PATH):
        self.struct_model = joblib.load(lr_path)
        self.sentiment_model = joblib.load(sentiment_path)
        self.sbert = SentenceTransformer('jhgan/ko-sroberta-multitask')
        print("HybridChurnPredictor 초기화 완료. 리소스 로드함.")

    def predict(self, struct_df: pd.DataFrame, user_text: str = None) -> Tuple[float, float, float]:
        if self.struct_model is None:
            return 0.0, 0.0, 0.0

        # 1. 정형 데이터 예측
        prob_struct = self.struct_model.predict_proba(struct_df)[0][1]

        # 2. 텍스트 감성 예측
        prob_text = 0.0
        if self.sentiment_model and self.sbert and user_text:
            try:
                text_vec = self.sbert.encode([user_text])
                prob_text = self.sentiment_model.predict_proba(text_vec)[0][1]
            except Exception as e:
                # 들여쓰기 수정
                print(f"감성 분석 오류: {e}")
        
        # 3. 하이브리드 결합
        if not user_text or (self.sentiment_model is None):
            final_prob = prob_struct
        else:
            final_prob = (prob_struct * WEIGHT_STRUCT) + (prob_text * WEIGHT_TEXT)

        return final_prob, prob_struct, prob_text

# ======================================================================
# 3. 전처리 및 예측 함수
# ======================================================================

def get_customer_features_by_id(user_id: str, df_cluster: pd.DataFrame) -> Dict[str, Any]:
    """
    입력한 user_id의 피처를 반환하는 함수
    """
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
    """
    고객 정보 전처리하는 함수
    """
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

def find_retained_neighbors(b_id: Any, df_cluster: pd.DataFrame) -> pd.DataFrame:
    """
    고객의 군집 내에서 이탈하지 않고 서비스를 계속 이용 중인 고객 찾는 함수

    Params
    --------
    b_id: 이탈 위험 고객과 유사한 고객의 id
    df_cluster

    Returns
    --------
    neighbors: 이탈하지 않은 고객 집합
    """
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
# 4. 캡슐화된 대조 분석기 클래스 (메인 클래스)
# ======================================================================

class ContrastiveAnalyzer:
    """모든 리소스를 로드하고, user_id와 text만으로 분석을 수행하는 클래스"""

    def __init__(self, paths: Dict[str, str] = RESOURCE_PATHS):
        # 1. 대조 분석용 리소스
        self.corpus_emb = joblib.load(paths['emb'])
        self.df_text = pd.read_csv(paths['text'])
        self.df_cluster = pd.read_csv(paths['cluster'])
        self.sbert_contrast = SentenceTransformer('jhgan/ko-sroberta-multitask')
        
        # 2. 예측용 리소스
        self.hybrid_predictor = HybridChurnPredictor(lr_path=paths['lr'], sentiment_path=paths['sentiment'])
        self.scaler = joblib.load(paths['scaler'])
            
        print("ContrastiveAnalyzer 초기화 완료.")

    def analyze_user(self, user_id: str, consult_text: str, user_features_dict: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        주어진 user_id와 상담 텍스트를 기반으로 이탈 예측 및 대조 분석을 수행한다.
        user_features_dict는 신규 고객 분석(NewUser) 시에만 사용된다.
        """
        try:
            # 1. 피처 데이터 준비: user_features_dict가 있으면 사용, 없으면 ID로 조회
            if user_features_dict:
                raw_features = user_features_dict
            else:
                raw_features = get_customer_features_by_id(user_id, self.df_cluster)
                
            if not raw_features:
                return {"role_model_pattern": "분석 불가 (ID 오류)", "insight": f"고객 ID {user_id}의 정보가 데이터에 없습니다.", "churn_probability": 0.0}
                
            A_df_raw = process_user_input_to_df(raw_features)
            A_df_scaled = A_df_raw.copy()
            
            # 스케일러 적용
            try:
                if self.scaler is not None:
                    cols_to_scale = SCALE_COLS_FOR_TRANSFORM
                    numerical_data = A_df_scaled[cols_to_scale]
                    scaled_data = self.scaler.transform(numerical_data)
                    A_df_scaled[cols_to_scale] = scaled_data
            except Exception as e:
                print(f"[경고] Scaler 적용 중 오류 발생: {e}. Raw Data로 예측 시도합니다.")
                A_df_scaled = A_df_raw 

            # 2. 하이브리드 이탈 예측
            final_prob, prob_struct, prob_text = self.hybrid_predictor.predict(A_df_scaled, consult_text) 
            churn_prob = final_prob
            
            if churn_prob <= 0.5:
                print(f"[대조 분석 생략]예측 이탈 확률 ({churn_prob:.2%})이 50% 이하이므로 대조 분석을 생략합니다.")
                return {
                    "role_model_pattern": "저위험군",
                    "insight": f"이탈 확률이 기준(50%) 이하입니다. (정형:{prob_struct:.2%}, 텍스트:{prob_text:.2%})",
                    "churn_probability": float(f"{churn_prob:.4f}")
                }
            
            # 3. 유사 고객(Role Model) 찾기 
            current_consult_text = consult_text if consult_text else None

            b_id, sim_score = find_most_similar_customer_B(current_consult_text, self.sbert_contrast, self.corpus_emb, self.df_text)
            
            if b_id is None:
                return {"role_model_pattern": "유사 사례 매칭 실패", "insight": "유사 텍스트를 찾았으나 ID 매칭 실패", "churn_probability": float(f"{churn_prob:.4f}")}

            # 4. 유사 군집 및 롤모델 찾기 
            neighbors = find_retained_neighbors(b_id, self.df_cluster)

            # 5. 차이점 분석 (제한 없는 모든 차이점 추천 로직)
            service_recommendations = []
            payment_recommendation = ""
            target_services = ['OnlineSecurity', 'OnlineBackup', 'TechSupport', 'UnlimitedData', 'PaperlessBilling']
            
            
            if raw_features:
                # 5-1. 서비스 가입 추천 (고객 A 미가입 & 유사 그룹 중 한 명이라도 가입)
                for col in target_services:
                    my_val = raw_features.get(col, '미가입')
                    
                    if col in neighbors.columns:
                        if str(my_val).lower() in ['no', '미가입', '0', 'false', 'nan']:
                            # 유사 그룹 중 한 명이라도 가입했는지 확인
                            has_neighbor_used = neighbors[col].apply(lambda x: str(x).lower() in ['yes', '1', 'true']).any()
                            if has_neighbor_used: 
                                service_recommendations.append(f"{col}") 

                # 5-2. 결제 수단 변경 추천 (고객 A와 다른 결제 수단이 유사 그룹에 등장하면 모두 추천)
                my_payment = VALUE_MAPPING.get(str(raw_features.get('PaymentMethod', '계좌이체')).strip(), str(raw_features.get('PaymentMethod', '계좌이체')).strip())
                
                if 'PaymentMethod' in neighbors.columns:
                    neighbor_payments = neighbors['PaymentMethod'].apply(lambda x: VALUE_MAPPING.get(str(x).strip(), str(x).strip())).unique()
                    
                    for payment in neighbor_payments:
                        if payment != my_payment:
                            payment_recommendation = f"{payment}" 
                            break # 첫 번째 차이나는 결제 수단만 추천

            # 6. 결과 출력 문자열 조합
            recommendation_parts = []
            if service_recommendations:
                recommendation_parts.extend(service_recommendations)
            if payment_recommendation:
                recommendation_parts.append(payment_recommendation)
            
            recommendations_str = ", ".join(recommendation_parts) 

            if not recommendations_str:
                return {
                    "role_model_pattern": "추가 추천 서비스가 없습니다.",
                    "insight": "유사한 만족 고객들은 현재 고객과 차이가 거의 없습니다. 이탈 위험 요인을 확인하세요. ",
                    "churn_probability": float(f"{churn_prob:.4f}")
                }
            return {
                "role_model_pattern": recommendations_str,
                "insight": f"이 고객과 유사한 만족 고객들은 {recommendations_str} 등을 이용 중입니다. 이를 제안하면 이탈 위험 감소에 도움이 될 수 있습니다.",
                "prob_struct": f"{prob_struct:.2%}",
                "prob_text": f"{prob_text:.2%}",
                "churn_probability": float(f"{churn_prob:.4f}")
            }

        except Exception as e:
            print(f"[대조분석] 로직 실행 중 에러: {e}")
            return {"role_model_pattern": "오류", "insight": str(e), "churn_probability": 0.0}

    def run_analysis(self, user_id: str, user_text: str, user_features_input: Dict[str, Any] = None):
        """테스트 래퍼 함수: user_id와 text만 입력받아 분석을 실행하고 결과를 출력"""
        
        result = self.analyze_user(user_id, user_text, user_features_dict=user_features_input)
        
        print(f"--- 1. 입력 처리 완료: ID={user_id}, Text={user_text}")
        print(f"--- 2. 이탈 예측 확률: {result.get('churn_probability', 0.0) * 100:.2f}%")
        print("\n3. 분석 결과:")
        print(f"    - Role Model Pattern: {result.get('role_model_pattern', '오류')}")
        print(f"    - Insight: {result.get('insight', '분석 중 오류 발생')}")
        
        return result.get('churn_probability', 0.0)
    
if __name__ == "__main__":
    print("=====================================================")
    print("          이탈 예측 및 대조 분석 모듈 ")
    print("=====================================================")
    
    # 1. 분석기 객체 생성
    analyzer = ContrastiveAnalyzer()
    
    # 2. 테스트 유저
    user_features_input = {
    "CustomerID": "0001",
    "Gender": "남자",
    "Age": 30,
    "Married": "Yes",
    "Dependents": "No",
    "Tenure_month": 1,
    "Monthly_charge": 110.00,
    "Sum_charge": 110.00,
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "TechSupport": "No",
    "UnlimitedData": "Yes",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "계좌이체",
    "Device Protection": "No",
    "Contract": "Month-to-Month",
    "StreamingTV": "No",
    "ConsultText": "엥"
}

    print("\n----------------------------------------------------")
    print(f"신규 고객 분석 (ID: {user_features_input['CustomerID']})")
    print("----------------------------------------------------")
    
    # 4. 실행
    result_prob = analyzer.run_analysis(
        user_id=user_features_input['CustomerID'],
        user_text=user_features_input['ConsultText'],
        user_features_input=user_features_input 
    )
    
    print("\n=====================================================")
    print(f" 최종 예측 확률: {result_prob * 100:.2f}%")
    print("=====================================================")