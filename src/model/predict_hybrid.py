import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import warnings

WEIGHT_STRUCT = 0.7
WEIGHT_TEXT = 0.3

# 파일 경로 (VS Code 프로젝트 폴더 구조 기준)
LR_MODEL_PATH = 'data/processed/lr_model.joblib'             # 정형 모델 파일
SENTIMENT_MODEL_DIR = 'data/processed/llm_sentiment_model'   # 감성 모델 폴더

#텔코 데이터 모델 학습에 사용된 컬럼 목록
MODEL_COLS = [
    # 기본 정보
    'Gender', 'Age', 'Married', 'Dependents', 'noDependents', 
    'Referrals', 'noReferrals', 'PaperlessBilling', 
    'OnlineSecurity', 'OnlineBackup', 'TechSupport', 'UnlimitedData', 
    'AvgDownloadGB', 'CustomerLTV', 'SatisScore', 
    'TotalExtraDataCharge', 'AvgRoamCharge', 'TotalRoamCharge', 
    'Tenure_month', 'Sum_charge', 'Monthly_charge', 'ServiceDuration', 
    
    # 파생 변수
    'CLTV_monthly', 'TotalOtherCharges', 'LTVPerSatis', 'Is_Manual_Payment', 
    
    # OHE (Drop First 적용됨)
    'PaymentMethod_신용카드', 'PaymentMethod_이체/메일확인', 
    'AgeGroup_30대', 'AgeGroup_40대', 'AgeGroup_50대', 
    'AgeGroup_60대', 'AgeGroup_70대', 'AgeGroup_80대'
]

# 값 매핑 사전
VALUE_MAPPING = {
    '남자': '남성', '남': '남성', 'male': '남성', 
    '여자': '여성', '여': '여성', 'female': '여성',
    '가입': 'Yes', '사용': 'Yes', '예': 'Yes', 'true': 'Yes', '1': 'Yes',
    '미가입': 'No', '미사용': 'No', '아니요': 'No', 'false': 'No', '0': 'No',
    '신용카드': '신용카드', '계좌이체': '계좌이체', 
    '이체': '이체/메일확인', '메일확인': '이체/메일확인', '이체/메일확인': '이체/메일확인'
}

# 단순 이진형 컬럼 (Gender, Dependents 등 제외)
BINARY_COLS = ['Married', 'PaperlessBilling', 'OnlineSecurity', 
               'OnlineBackup', 'TechSupport', 'UnlimitedData']

class DataPreprocessor:
    @staticmethod
    def process(A_features_raw):
        """사용자 입력을 받아 학습된 모델의 컬럼 구조(34개)로 변환"""
        
        # 1. 빈 틀 생성 (모든 값 0으로 초기화)
        feature_data = {col: 0 for col in MODEL_COLS}

        # 2. 입력 데이터 매핑
        for raw_key, raw_value in A_features_raw.items():
            clean_val = str(raw_value).strip()

            # (A) 성별 (Gender -> 1/0)
            if raw_key == 'Gender':
                mapped = VALUE_MAPPING.get(clean_val, clean_val)
                # 남성이면 1, 여성이면 0
                feature_data['Gender'] = 1 if mapped in ['남성', 'Male'] else 0
            
            # (B) 단순 이진형 (Yes -> 1, No -> 0)
            elif raw_key in BINARY_COLS:
                mapped = VALUE_MAPPING.get(clean_val.lower(), clean_val)
                feature_data[raw_key] = 1 if mapped.lower() in ['yes', '1'] else 0

            # (C) Dependents & Referrals (대칭 변수 처리)
            # Dependents가 1이면 noDependents는 0이어야 함
            elif raw_key == 'Dependents':
                mapped = VALUE_MAPPING.get(clean_val.lower(), clean_val)
                is_yes = 1 if mapped.lower() in ['yes', '1'] else 0
                feature_data['Dependents'] = is_yes
                feature_data['noDependents'] = 1 - is_yes # 반대값
            
            elif raw_key == 'Referrals':
                mapped = VALUE_MAPPING.get(clean_val.lower(), clean_val)
                is_yes = 1 if mapped.lower() in ['yes', '1'] else 0
                feature_data['Referrals'] = is_yes
                feature_data['noReferrals'] = 1 - is_yes # 반대값

            # (D) 수치형 (Age 등)
            elif raw_key in feature_data:
                try: feature_data[raw_key] = float(clean_val)
                except: pass

            # (E) Age -> AgeGroup OHE (Baseline: 20대)
            # 20대면 아무 컬럼도 1이 되지 않음 (All 0)
            if raw_key == 'Age':
                try:
                    age = int(float(clean_val))
                    feature_data['Age'] = age
                    decade = (age // 10) * 10
                    target_col = f"AgeGroup_{decade}대"
                    if target_col in feature_data:
                        feature_data[target_col] = 1
                except: pass
                
            # (F) PaymentMethod OHE (Baseline: 계좌이체)
            # 계좌이체면 아무 컬럼도 1이 되지 않음 (All 0)
            if raw_key == 'PaymentMethod':
                mapped_pm = VALUE_MAPPING.get(clean_val, clean_val)
                target_col = f"PaymentMethod_{mapped_pm}"
                if target_col in feature_data:
                    feature_data[target_col] = 1

        # [3] 파생변수 자동 계산 (모델 필수값)
        feature_data['TotalOtherCharges'] = feature_data.get('TotalExtraDataCharge', 0) + feature_data.get('TotalRoamCharge', 0)
        
        ltv = feature_data.get('CustomerLTV', 0)
        dur = feature_data.get('ServiceDuration', 1) # 0 나누기 방지
        if dur == 0: dur = 1
        feature_data['CLTV_monthly'] = ltv / dur
        
        satis = feature_data.get('SatisScore', 1)
        if satis == 0: satis = 1
        feature_data['LTVPerSatis'] = ltv / satis
        
        # Is_Manual_Payment: 신용카드(1) 혹은 이체/메일확인(1)이면 수동
        # 계좌이체(둘 다 0)면 자동(0)
        is_manual = 0
        if feature_data.get('PaymentMethod_신용카드') == 1 or feature_data.get('PaymentMethod_이체/메일확인') == 1:
            is_manual = 1
        feature_data['Is_Manual_Payment'] = is_manual

        # DataFrame 변환 (반드시 학습 컬럼 순서 유지)
        return pd.DataFrame([feature_data], columns=MODEL_COLS)
    

#하이브리드 예측 클래스 (LR + LLM)
class HybridChurnPredictor:
    def __init__(self):
        self.lr_model = None
        self.tokenizer = None
        self.sentiment_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_models()

    def _load_models(self):
        print("📦 [Hybrid System] 모델 로딩 중...")
        
        # 1. 정형 모델 로드
        if os.path.exists(LR_MODEL_PATH):
            self.lr_model = joblib.load(LR_MODEL_PATH)
            print("  ✅ 정형 예측 모델(LR) 로드 완료")
        else:
            print(f"  ❌ 정형 모델 파일 없음: {LR_MODEL_PATH}")

        # 2. 감성 모델 로드 (LLM)
        if os.path.exists(SENTIMENT_MODEL_DIR):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_DIR)
                self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_DIR)
                self.sentiment_model.to(self.device)
                print("  ✅ 감성 분석 모델(LLM) 로드 완료")
            except Exception as e:
                print(f"  ❌ 감성 모델 로드 실패: {e}")
        else:
            print(f"  ⚠️ 감성 모델 폴더 없음: {SENTIMENT_MODEL_DIR}")

    def predict_proba(self, user_features: dict, user_text: str):
        """
        최종 이탈 확률 예측 (정형 + 비정형)
        """
        # 1. 정형 데이터 예측
        prob_struct = 0.0
        if self.lr_model:
            try:
                # 전처리 (Dictionary -> DataFrame)
                df_input = DataPreprocessor.process(user_features)
                # 확률 예측 (1: 이탈)
                prob_struct = self.lr_model.predict_proba(df_input)[0][1]
            except Exception as e:
                print(f"  ⚠️ 정형 예측 중 오류: {e}")

        # 2. 감성 분석 예측
        prob_text = 0.0
        if self.sentiment_model and user_text:
            try:
                inputs = self.tokenizer(
                    user_text, return_tensors="pt", truncation=True, max_length=128
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.sentiment_model(**inputs)
                    probs = F.softmax(outputs.logits, dim=-1)
                    
                # 1번 클래스(부정/이탈위험) 확률
                prob_text = probs[0][1].item()
            except Exception as e:
                print(f"  ⚠️ 감성 분석 중 오류: {e}")
        
        # 3. 하이브리드 결합
        if not user_text:
            final_prob = prob_struct
        else:
            final_prob = (prob_struct * WEIGHT_STRUCT) + (prob_text * WEIGHT_TEXT)

        return final_prob, prob_struct, prob_text


# ==========================================
# 4. 실행 테스트
# ==========================================
if __name__ == "__main__":
    # 예측기 초기화
    predictor = HybridChurnPredictor()
    
    # 테스트 데이터
    test_text = "요금이 너무 비싸서 화가 납니다. 해지할래요."
    test_features = {
        'Gender': '남자', 
        'Age': 24,  # -> AgeGroup 모두 0 (20대 Baseline)
        'PaymentMethod': '계좌이체', # -> PaymentMethod 모두 0 (Baseline)
        'Monthly_charge': 95000,
        'CustomerLTV': 5000,
        'ServiceDuration': 12,
        'OnlineSecurity': 'No'
    }
    
    # 예측 실행
    final, p_struct, p_text = predictor.predict_proba(test_features, test_text)
    
    print("\n" + "="*40)
    print(f"📊 [하이브리드 이탈 예측 결과]")
    print(f"  - 정형 데이터(패턴) 확률: {p_struct:.2%} (가중치 {WEIGHT_STRUCT})")
    print(f"  - 상담 텍스트(감성) 확률: {p_text:.2%} (가중치 {WEIGHT_TEXT})")
    print("-" * 40)
    print(f"  👉 최종 이탈 확률: {final:.2%}")
    print("="*40)