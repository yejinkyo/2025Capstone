import json
import os
import pandas as pd

# 분리한 분석 모듈 import
from model.rec_v2 import init_shap_model, analyze_customer_shap
from model.rec_v1 import init_contrastive_resources, perform_contrastive_analysis_for_user

# 파일 경로 설정
DATA_FILE = "data/user_analysis.json"
SHAP_DATA_PATH = "data/raw/telco2.csv" # SHAP용 원본 데이터

# =========================================================
# 1. 서버 시작 시 리소스 로딩 
# =========================================================

# SHAP 모델 로드
shap_model, shap_df = init_shap_model(SHAP_DATA_PATH)

# 대조분석 리소스 로드
contrast_resources = init_contrastive_resources()

print("시스템 초기화 완료")

# =========================================================
# 2. 분석 및 업데이트 함수 (UI에서 호출)
# =========================================================
def update_customer_analysis(user_id, consultation_text=""):
    """
    [핵심 로직]
    1. SHAP 분석 수행 -> 이탈 원인(Why) 도출
    2. 대조 분석 수행 -> 성공 해법(How) 도출
    3. user_analysis.json 파일 업데이트
    """
    print(f"\n고객 ID: {user_id}")

    # -----------------------
    # 1. SHAP 분석 (Why)
    # -----------------------
    shap_result = analyze_customer_shap(shap_model, shap_df, user_id)
    
    shap_data = {
        "cause": shap_result['pain_point'],
        "detail": shap_result['detail']
    }
    user_info = f"이탈 위험도 {shap_result['churn_prob']:.1%}"

    # -----------------------
    # 2. 대조 분석 (How)
    # -----------------------
    # 상담 텍스트가 없으면 기본값 처리
    text_input = consultation_text if consultation_text else ""
    
    contrast_result = perform_contrastive_analysis_for_user(user_id, text_input, contrast_resources)
    
    contrast_data = {
        "role_model_pattern": contrast_result['role_model_pattern'],
        "insight": contrast_result['insight']
    }

    # -----------------------
    # 3. JSON 저장
    # -----------------------
    new_entry = {
        "info": {
            "desc": user_info,
            "history": f"상담 내용: {text_input}"
        },
        "shap": shap_data,
        "contrastive": contrast_data
    }

    # 기존 파일 읽기
    current_data = {}
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                current_data = json.load(f)
        except:
            current_data = {}

    # 데이터 업데이트
    current_data[str(user_id)] = new_entry

    # 파일 쓰기
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(current_data, f, ensure_ascii=False, indent=2)

    print(f"[저장 완료] '{DATA_FILE}' 업데이트 됨.")
    return True

# 테스트 실행
if __name__ == "__main__":
    test_id = "0280-XJGEX"
    
    update_customer_analysis(test_id)