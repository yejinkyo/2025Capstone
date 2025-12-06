import pandas as pd
import json

from src.model.rec_v2 import init_shap_model, adapt_telco_to_shap, analyze_customer_shap
from src.model.rec_v1 import ContrastiveAnalyzer

# ====================================================
# JSON 통합 함수
# ====================================================
def create_llm_payload(user_info, shap_result, contrast_result):
    # 추천 아이템 병합 및 중복 제거
    shap_items = shap_result.get('top_recommendations', [])
    contrast_items = []
    
    # 대조분석 결과 파싱
    c_pattern = contrast_result.get('role_model_pattern', '')
    if isinstance(c_pattern, list):
        contrast_items = c_pattern
    elif isinstance(c_pattern, str) and c_pattern:
        contrast_items = [item.strip() for item in c_pattern.split(',')]

    all_recommendations = list(set(shap_items + contrast_items))

    # (2) JSON 구조 생성
    payload = {
        "customer_profile": {
            "customer_id": user_info.get('CustomerId', 'Unknown'),
            "gender": user_info.get('Gender', 'Unknown'),
            "tenure_month": user_info.get('Tenure_month', 0),
            "monthly_charge": user_info.get('Monthly_charge', 0),
            "age": user_info.get('Age', 'Unknown'),
            "streaming_tv": user_info.get('StreamingTV', 'Unknown'),
            "payment_method": user_info.get('PaymentMethod', 'Unknown'),
            "online_security": user_info.get('OnlineSecurity', 'Unknown')
        },
        "analysis_results": {
            "shap_analysis": {
                "churn_probability": f"{shap_result.get('churn_prob', 0):.1%}",
                "risk_level": shap_result.get('pain_point', ''),
                "recommended_actions": shap_items,
                "reasoning": shap_result.get('detail', '')
            },
            "contrastive_analysis": {
                "insight_message": contrast_result.get('insight', ''),
                "recommended_services": contrast_items,
                "churn_probability": contrast_result.get('churn_probability', '')
            }
        }
    }
    
    return json.dumps(payload, ensure_ascii=False, indent=4)

# 대조분석 + SHAP 추천 결과 생성 함수
def generate_analysis_result(user_id, consult_text, shap_resources, analyzer_instance, user_features_input):
    """
    """
    # 1. resource 언팩
    shap_model, shap_df = shap_resources

    # 2. 유저 정보 찾기 (대조분석)
    analyzer = analyzer_instance 
    df_cluster = analyzer.df_cluster

    id_col = 'CustomerID'
    if id_col not in df_cluster.columns:
        for c in ['CustomerId', 'id']:
            if c in df_cluster.columns:
                id_col = c
                break
    
    # 해당 id의 행 검색
    user_row = df_cluster[df_cluster[id_col].astype(str) == str(user_id)]

    # 새로운 유저로 분석 
    if not user_row.empty:
        # DB에 존재하는 기존 고객
        target_user_dict = user_row.iloc[0].to_dict()
    elif user_features_input:
        # DB에 없는 신규 고객 
        target_user_dict = user_features_input
    else:
        # Case C: DB에도 없고 입력도 없음 -> 에러
        return json.dumps({"error": f"ID {user_id} not found"}, indent=4)

    # 3. 대조 분석 실행
    contrast_result = analyzer.analyze_user(
        user_id=str(user_id),
        consult_text=consult_text,
        user_features_dict=user_features_input
    )

    # 4. SHAP 분석 실행
    shap_input_df = adapt_telco_to_shap(target_user_dict)
    shap_result = analyze_customer_shap(
        shap_model, 
        shap_df, 
        custom_data=shap_input_df
        )
    
    # 5. json 생성
    final_json_str = create_llm_payload(target_user_dict, shap_result, contrast_result)

    return final_json_str

# ====================================================
# 메인 실행 
# ====================================================
if __name__ == "__main__":
    # 리소스 로드 
    shap_model, shap_df_origin = init_shap_model('data/raw/telco2.csv') 

    # ---------------------------------------------------------
    # 테스트용 가상 고객 데이터 
    # ---------------------------------------------------------

    target_user = {
        'CustomerId': 'NewUser_Example_9999', # 임의의 ID
        'Gender': '남자', 
        'Age': 30,  
        'Married': 'Yes',
        'Dependents': 'Yes',
        'Referrals': 'No',
        'PaperlessBilling': 'Yes',
        'OnlineSecurity': '미가입',    # 추천 대상 후보
        'OnlineBackup': '미가입',      # 추천 대상 후보
        'TechSupport': '미가입',       # 추천 대상 후보
        'UnlimitedData': 'No',         # 무제한 데이터 미가입
        'StreamingTV': '미가입',
        'PaymentMethod': '계좌이체',    # 결제 수단 추천 후보
        'AvgDownloadGB': 5.5,
        'CustomerLTV': 500,
        'SatisScore': 2,
        'TotalExtraDataCharge': 50,
        'AvgRoamCharge': 10,
        'TotalRoamCharge': 100,
        'Tenure_month': 15,
        'Sum_charge': 1500,
        'Monthly_charge': 100,
        'ServiceDuration': 10,
        'Contract': 'Yes'
    }

    user_consult_text = "요금제가 너무 비싸고 느려서 해지하고 싶어요."

    # SHAP 리소스 튜플로 묶기
    shap_resources_tuple = (shap_model, shap_df_origin)

    # 대조 분석기 인스턴스 생성
    analyzer = ContrastiveAnalyzer()

    # 추천 결과 생성 
    result = generate_analysis_result(
        user_id=target_user['CustomerId'],
        consult_text=user_consult_text,
        user_features_input=target_user,
        shap_resources=shap_resources_tuple, # 튜플로 전달
        analyzer_instance=analyzer            # 생성된 인스턴스 전달
    )

    # ---------------------------------------------------------
    # LLM용 JSON Payload 생성
    # ---------------------------------------------------------

    print("\n" + "="*50)
    print("[최종 생성된 JSON 데이터 (LLM 입력용)]")
    print("="*50)
    print(result)