import pandas as pd
import json

from src.model.rec_v2 import init_shap_model, adapt_telco_to_shap, analyze_customer_shap
from src.model.rec_v1 import init_contrastive_resources, perform_contrastive_analysis_for_user

# ====================================================
# JSON 통합 함수
# ====================================================
def create_llm_payload(user_info, shap_result, contrast_result):
    # (1) 추천 아이템 병합 및 중복 제거
    shap_items = shap_result.get('top_recommendations', [])
    contrast_items = []
    
    # 대조분석 결과 파싱 (문자열인 경우 리스트로 변환)
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
            "monthly_charge": user_info.get('Monthly_charge', 0)
        },
        "analysis_results": {
            "shap_analysis": {
                "churn_probability": f"{shap_result.get('churn_prob', 0):.1%}",
                "risk_level": shap_result.get('pain_point', '알 수 없음'),
                "recommended_actions": shap_items,
                "reasoning": shap_result.get('detail', '')
            },
            "contrastive_analysis": {
                "insight_message": contrast_result.get('insight', ''),
                "recommended_services": contrast_items
            }
        }
    }
    
    return json.dumps(payload, ensure_ascii=False, indent=4)

def generate_analysis_result(user_id, consult_text, shap_resources, contrast_resources):
    """
    Gradio에서 버튼 클릭시 실행됨.
    """
    # 1. resource 언팩
    shap_model, shap_df = shap_resources

    # 2. 유저 정보 찾기 (대조분석 데이터에서 검색)
    _, _, _, _, df_cluster, _ = contrast_resources

    id_col = 'CustomerID'
    if id_col not in df_cluster.columns:
        for c in ['CustomerId', 'id']:
            id_col = c
            break
    
    # 해당 id의 행 검색
    user_row = df_cluster[df_cluster[id_col].astype(str) == str(user_id)]

    if user_row.empty:
        return json.dumps({"error": f"ID {user_id} not found"}, indent=4)
    
    target_user_dict = user_row.iloc[0].to_dict()

    # 3. 대조 분석 실행
    contrast_result = perform_contrastive_analysis_for_user(
        user_id=str(user_id),
        consult_text=consult_text,
        resources=contrast_resources
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
    contrast_resources = init_contrastive_resources()

    # ---------------------------------------------------------
    # 테스트용 가상 고객 데이터 
    # ---------------------------------------------------------
    target_user = {
        'CustomerId': 'User-001',
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
    
    # 고객 상담
    user_consult_text = "요금이 좀 비싼 것 같고, 폰이 자주 고장나서 걱정이에요."

    # ---------------------------------------------------------
    # 1. 대조 분석 실행
    # ---------------------------------------------------------
    user_df_contrast = pd.DataFrame([target_user])
    
    contrast_result = perform_contrastive_analysis_for_user(
        user_id=target_user['CustomerId'],
        consult_text=user_consult_text,
        resources=contrast_resources,
        user_df=user_df_contrast 
    )

    # ---------------------------------------------------------
    # 2. SHAP 분석 실행 
    # ---------------------------------------------------------
    # 데이터 변환
    shap_input_df = adapt_telco_to_shap(target_user)
    
    # 분석 
    shap_result = analyze_customer_shap(
        shap_model, 
        shap_df_origin, 
        custom_data=shap_input_df
    )

    # ---------------------------------------------------------
    # LLM용 JSON Payload 생성
    # ---------------------------------------------------------
    final_json = create_llm_payload(target_user, shap_result, contrast_result)

    print("\n" + "="*50)
    print("[최종 생성된 JSON 데이터 (LLM 입력용)]")
    print("="*50)
    print(final_json)