from sklearn.model_selection import train_test_split
import pandas as pd

def split_data(df, target_col, test_size=0.2, random_state=42):
    """
    데이터를 train/test로 분리하는 함수

    Parameters
    -----------
    df: pandas.DataFrame
        전처리 완료된 데이터프레임
    target_col: str
        타겟 변수 이름
    test_size: float
        테스트 데이터 비율
    random_state: int
        랜덤 시드 값
    
    Returns
    -----------
    X_train, X_test, y_train, y_test: pandas.DataFrame, pandas.Series
    """

    # 입력(X), 타겟(y) 분리
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )

    X_train = X_train.drop(columns=['Unnamed: 0', 'CustomerId', 'StartDate', 'EndDate', 'EndDateTmp', 'ChurnCategory', 'ChurnReason', 'ChurnScore', 'kmeans_cluster_id', "cluster_name", "ChurnLabel"], errors='ignore')
    X_test = X_test.drop(columns=['Unnamed: 0', 'CustomerId', 'StartDate', 'EndDate', 'EndDateTmp', 'ChurnCategory', 'ChurnReason', 'ChurnScore', 'kmeans_cluster_id', "cluster_name", "ChurnLabel"], errors='ignore')


    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = pd.read_csv("data/processed/telco_cleaned_data.csv")  # 전처리 완료된 파일
    X_train, X_test, y_train, y_test = split_data(df, target_col='ChurnLabel')

    # 나눈 데이터 저장
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)