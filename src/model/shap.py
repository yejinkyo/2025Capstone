import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
import shap.maskers

def load_model(model_path):
    """
    저장된 이탈 확률 예측 모델 불러오기
    """
    model = joblib.load(model_path)
    return model

def recommend_services(model, X_new, target_service_features, X_train_background=None, show_plots=False):
    """
    불러온 모델을 사용해 새로운 데이터에 대한 서비스 추천(중요도 분석) 수행

    Params
    -------
    model: 학습된 로지스틱 회귀 모델
    X_new: 새로운 고객 데이터
    target_service_features: 분석 대상 서비스 피처 목록
    X_train_background: SHAP 배경 데이터로 사용할 학습 데이터
    show_plots: SHAP 그래프 표시 여부

    Returns
    -------
    predictions: 이탈 확률
    shap_values: SAHP 값
    """

    # 1. 데이터 필터링: 분석 대상 서비스 피처만 선택
    existing_features = [col for col in target_service_features if col in X_new.columns]

    X_new_subset = X_new[existing_features].copy()
    print(f"분석 대상 피처: {existing_features}")

    # 2. 이탈 확률 예측
    predictions = model.predict_proba(X_new_subset)[:, 1]

    # 3. SHAP 분석
    print("n\=== SHAP Analysis Recommendations ===")

    # 배경 데이터 설정
    if X_train_background is not None:
        background_data = shap.maskers.Independent(X_train_background[existing_features], max_samples=100)
    else:
        background_data = shap.maskers.Independent(X_new_subset, max_samples=100)
    
    explainer = shap.LinearExplainer(model, background_data)
    shap_values = explainer(X_new_subset)

    if show_plots:
        print("\n=== 그래프  시각화 ===")
        plt.figure()
        shap.plots.beeswarm(shap_values, max_display=len(existing_features))
        plt.tight_layout
        plt.show()

        try:
            shap.initjs()
            print("\n === 첫번째 고객에 대한 SHAP Force Plot ===")
            display(shap.plots.force(shap_values[0]))
        except:
            print("\n 주피터 환경에서만 실행 가능")
    
    return predictions, shap_values


if __name__ == "__main__":
    from sklearn.datasets import make_classification

    # 1. 모델 경로 및 데이터 경로 설정
    MODEL_PATH = 'data/processed/lr_model_services_only.joblib'

    # 테스트용 가상 데이터 생성
    X_dummy, _ = make_classification(n_samples=50, n_features=20, random_state=100)
    dummy_cols = ['PaperlessBilling', 'OnlineSecurity', 'OnlineBackup', 'TechSupport', 'UnlimitedData'] + [f'Other_{i}' for i in range(15)]
    X_new = pd.DataFrame(X_dummy, columns=dummy_cols)

    # 2. 모델 불러오기
    loaded_model = load_model(MODEL_PATH)

    if loaded_model is not None:
        # 3. 분석 대상 서비스 피처 목록 정의
        target_service_features = [
            'PaperlessBilling',
            'OnlineSecurity',
            'OnlineBackup',
            'TechSupport',
            'UnlimitedData'
        ]

        # 4. SHAP 배경 데이터 로드
        X_train_background = pd.read_csv("data/processed/X_train_preprocessed.csv", encoding="cp949")
        
        # 5. 추천 함수 실행
        predictions, shap_values = recommend_services(
            loaded_model,
            X_new,
            target_service_features,
            X_train_background,
            show_plots=True
        )

        if predictions is not None:
            print("\n === 추천 결과 예시 ===")
            result_df = pd.DataFrame({
                '이탈 확률': predictions,
                '주요 위험 요인': X_new[target_service_features].columns[np.argmax(shap_values.values, axis=1)],
                '주요 방어 요인': X_new[target_service_features].columns[np.argmin(shap_values.values, axis=1)]

            })
            print(result_df.head())

            # 개별 고객의 SHAP 값 확인
            print(pd.Series(shap_values[0].values, index=target_service_features))
        
        else:
            print("모델 로드 오류")