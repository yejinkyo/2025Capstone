from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

def train_and_evaluate(X_train, y_train, X_test, y_test, model=None):
    """
    1. SMOTE 적용 -> 2. 모델 학습 -> 3. 예측 -> 4. 평가
    """
    if model is None:
        # max_iter를 늘려서 수렴 경고 방지
        model = LogisticRegression(max_iter=2000, random_state=42)

    print(f"Training model with {X_train.shape[1]} features...")

    # 1) SMOTE 적용
    # 주의: 피처 수가 매우 적을 때 SMOTE가 데이터 분포를 왜곡할 수도 있으나,
    # 여기서는 기존 프로세스를 유지합니다.
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    # 2) 모델 학습
    model.fit(X_train_sm, y_train_sm)

    # 3) 예측
    y_pred = model.predict(X_test)

    # 4) 평가
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return model, y_pred

def get_feature_importance(model, feature_names, top_n=None):
    """
    모델의 회귀 계수(coefficient) 기반 feature importance 반환
    """
    importance = model.coef_[0]

    # feature_names와 importance 개수 일치 확인
    if len(feature_names) != len(importance):
         # 데이터 전처리 과정에서 인코딩 등으로 칼럼 수가 변했을 경우를 대비한 예외처리
         print(f"Warning: Feature names count ({len(feature_names)}) does not match importance count ({len(importance)}).")
         # 임시로 importance 길이만큼만 사용 (실제 데이터에 맞게 수정 필요)
         feature_names = feature_names[:len(importance)]

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    
    # 절대값 기준으로 정렬하여 영향력이 큰 순서대로 보기
    importance_df['abs_importance'] = importance_df['importance'].abs()
    importance_df = importance_df.sort_values(by='abs_importance', ascending=False)

    if top_n:
        importance_df = importance_df.head(top_n)

    print(f"\n=== Feature Importances (Coefficient Based) ===")
    # 보기 좋게 출력
    print(importance_df[['feature', 'importance']].to_markdown(index=False))

    return importance_df

def analyze_with_shap(model, X_train, X_test, feature_names):
    """
    모델에 대해 SHAP 값 계산 및 시각화
    """
    print("\n=== SHAP Analysis (Subset) ===")

    # 1. SHAP Explainer 생성 (선형 모델 전용)
    # 배경 데이터로 학습 데이터의 일부를 사용 (속도 개선)
    background_data = shap.maskers.Independent(X_train, max_samples=100)
    explainer = shap.LinearExplainer(model, background_data)

    # 2. SHAP 값 계산
    print("SHAP 값 계산 중...")
    shap_values = explainer(X_test)

    # 3. SHAP 요약 차트 시각화
    print("\n=== 그래프 시각화 중 ===")
    plt.figure()
    # beeswarm plot이 요약 정보를 보기 가장 좋습니다.
    shap.plots.beeswarm(shap_values, max_display=len(feature_names))
    plt.title("SHAP Summary Plot (Subset Features)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return shap_values, explainer

# =========================================
# 메인 실행 블록
# =========================================
if __name__ == "__main__":
    # 1. 데이터 로드
    X_train_full = pd.read_csv("data/processed/X_train_preprocessed.csv", encoding="cp949")
    y_train = pd.read_csv("data/processed/y_train_preprocessed.csv").values.ravel()
    X_test_full = pd.read_csv("data/processed/X_test_preprocessed.csv")
    y_test = pd.read_csv("data/processed/y_test_preprocessed.csv").values.ravel()

    target_service_features = [
        'PaperlessBilling',
        'OnlineSecurity',
        'OnlineBackup',
        'TechSupport',
        'UnlimitedData'
    ]

    print(f"\n분석 대상 피처: {target_service_features}")

    # 3. 데이터 필터링: 해당 칼럼만 선택
    existing_features = [col for col in target_service_features if col in X_train_full.columns]

    if len(existing_features) != len(target_service_features):
        print("주의: 요청한 피처 중 일부가 데이터에 존재하지 않습니다.")
        print(f"찾은 피처: {existing_features}")
        missing = set(target_service_features) - set(existing_features)
        print(f"없는 피처: {missing}")

    if not existing_features:
        print("오류: 분석할 피처가 없습니다. 칼럼 이름을 확인해주세요.")
        exit()

    # 선택된 피처로만 구성된 새 데이터프레임 생성
    X_train_subset = X_train_full[existing_features].copy()
    X_test_subset = X_test_full[existing_features].copy()


    # 4. 모델 학습 (선택된 피처만 사용)
    # 이 모델은 오직 이 5가지 서비스 피처만 보고 이탈을 예측합니다.
    subset_model, y_pred_subset = train_and_evaluate(
        X_train_subset, y_train, X_test_subset, y_test
    )

    # 5. 회귀 계수 기반 중요도 확인
    # 이 모델 내에서의 상대적 중요도를 보여줍니다.
    # 양수(+)는 이탈 위험 증가, 음수(-)는 이탈 위험 감소(방어)를 의미합니다.
    get_feature_importance(subset_model, existing_features)


    # 6. SHAP 분석 실행
    # 선택된 피처 데이터와 피처 이름을 넘겨줍니다.
    shap_values, explainer = analyze_with_shap(
        subset_model, X_train_subset, X_test_subset, existing_features
    )

    # 모델 저장 (선택 사항)
    joblib.dump(subset_model, 'data/processed/lr_model_services_only.joblib')
    print("\n분석 완료.")