from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import joblib

def train_and_evaluate(X_train, y_train, X_test, y_test, model=None):
    """
    1. SMOTE 적용
    2. 모델 학습
    3. 예측
    4. 평가

    Params
    -------
    X_train, y_train: 훈련 데이터
    X_test, y_test: 테스트 데이터
    model: 사용할 모델 (기본값: LogisticRegression)

    Returns
    -------
    model: 학습된 모델
    y_pred: 예측 결과
    """
    if model is None:
        model = LogisticRegression(max_iter=1000, random_state=42)

    # 1) SMOTE 적용
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

def get_feature_importance(model, feature_names, top_n=10):
    """
    모델의 feature importance를 반환하는 함수

    Params
    -------
    model: 학습된 모델
    feature_names: 피처 이름 리스트
    top_n: 상위 N개의 피처 중요도 반환 (기본값: 10)

    Returns
    --------
    importance_df: 중요도 df
    """
    importance = model.coef_[0]

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values(by='importance', ascending=False).head(top_n)

    print("Top Feature Importances:".format(top_n))
    print(importance_df)

    return importance_df


if __name__ == "__main__":
    X_train = pd.read_csv("data/processed/X_train_preprocessed.csv")
    y_train = pd.read_csv("data/processed/y_train_preprocessed.csv").values.ravel()
    X_test = pd.read_csv("data/processed/X_test_preprocessed.csv")
    y_test = pd.read_csv("data/processed/y_test_preprocessed.csv").values.ravel()

    model, y_pred = train_and_evaluate(X_train, y_train, X_test, y_test)

    fi = get_feature_importance(model, X_train.columns.tolist(), top_n=20)

    joblib.dump(model, 'lr_model.joblib')