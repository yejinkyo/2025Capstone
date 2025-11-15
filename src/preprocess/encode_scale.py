import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

def preprocess_data(X_train, X_test, y_train, y_test):
    # Label Encoding
    label_cols = ['Gender', 'Married', 'Dependents', 'Referrals', 'PaperlessBilling',
                  'OnlineSecurity', 'OnlineBackup', 'TechSupport', 'UnlimitedData']
    for col in label_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
    
    le_target = LabelEncoder()
    y_train = le_target.fit_transform(y_train)
    y_test = le_target.transform(y_test)

    # One-Hot Encoding
    dummy_cols = ['PaymentMethod', 'AgeGroup']
    X_train = pd.get_dummies(X_train, columns=dummy_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=dummy_cols, drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    X_train = X_train.astype(int, errors='ignore')
    X_test = X_test.astype(int, errors='ignore')

    # Standardization
    scale_cols = ['Age', 'AvgDownloadGB', 'CustomerLTV', 'TotalExtraDataCharge',
                  'AvgRoamCharge', 'TotalRoamCharge', 
                  'noReferrals', 'noDependents', 'SatisScore', 'Tenure_month', 'Sum_charge', 'Monthly_charge', 'ServiceDuration',
                  'CLTV_monthly', 'TotalOtherCharges', 'LTVPerSatis', 'Is_Manual_Payment']
    scaler = StandardScaler()
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

    X_train, X_test, y_train, y_test = preprocess_data(X_train, X_test, y_train, y_test)

    X_train.to_csv("data/processed/X_train_preprocessed.csv", index=False)
    X_test.to_csv("data/processed/X_test_preprocessed.csv", index=False)
    pd.DataFrame(y_train, columns=['ChurnLabel']).to_csv("data/processed/y_train_preprocessed.csv", index=False)
    pd.DataFrame(y_test, columns=['ChurnLabel']).to_csv("data/processed/y_test_preprocessed.csv", index=False)