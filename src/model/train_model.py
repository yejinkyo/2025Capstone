from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd


if __name__ == "__main__":
    X_train = pd.read_csv("data/processed/X_train_preprocessed.csv")
    y_train = pd.read_csv("data/processed/y_train_preprocessed.csv").values.ravel()
    X_test = pd.read_csv("data/processed/X_test_preprocessed.csv")
    y_test = pd.read_csv("data/processed/y_test_preprocessed.csv").values.ravel()

    # SMOTE oversampling
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    # 학습
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_sm, y_train_sm)

    # 평가
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))