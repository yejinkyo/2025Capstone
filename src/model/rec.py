import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import SVD
from surprise import get_dataset_dir 
import numpy as np

def generate_recommendations(df):
    """
    고객 데이터 기반 추천 서비스 생성 (Rule-based + 협업 필터링)
    - cluster_name, AvgDownloadGB, SatisScore, ChurnScore, PaperlessBilling, PaymentMethod, Married, Dependents 활용
    - 과거 서비스 선호 데이터를 기반으로 협업 필터링 추천 추가
    """
    # ------------------
    # 1️. Rule-based 추천
    # ------------------
    
    cluster_services_map = {
        '표준 단기 고객 (월정액)': ['Standard Plan'],
        '알뜰형 장기 고객 (2년 약정, 저CLTV)': ['Budget Plan'],
        '기술선호형 중장기 고객 (월정액)': ['Tech Plan'],
        '초단기 신규 고객 (최대 이탈 위험군)': ['Trial Plan'],
        '표준 중기 고객 (1년 약정)': ['Standard Plan'],
        '고가치 장기 고객 (월정액, 고CLTV)': ['Premium Plan'],
        # 필요한 군집 모두 추가
    }

    df['RecommendedServices'] = df['cluster_name'].apply(lambda x: cluster_services_map.get(x, []))
    

    # 1. 사용량 기반 추천
    df['RecommendedServices'] = df.apply(
        lambda row: row['RecommendedServices'] + ['UnlimitedData'] if row['AvgDownloadGB'] > 20 else row['RecommendedServices'],
        axis=1
    )

    # 2. 만족도 기반 추천
    df['RecommendedServices'] = df.apply(
        lambda row: row['RecommendedServices'] + ['TechSupport'] if row['SatisScore'] < 3 else row['RecommendedServices'],
        axis=1
    )

    # 3. 이탈 위험 기반 추천
    df['RecommendedServices'] = df.apply(
        lambda row: row['RecommendedServices'] + ['OnlineBackup'] if row['ChurnScore'] > 60 else row['RecommendedServices'],
        axis=1
    )

    # 4. 디지털 선호 기반 추천
    df['RecommendedServices'] = df.apply(
        lambda row: list(set(row['RecommendedServices'] + ['TechSupport'])) 
        if (row['PaperlessBilling'] == 'Yes' and row['PaymentMethod'] == 'Electronic check') 
        else row['RecommendedServices'],
        axis=1
    )

    # 5. 가족 기반 추천
    df['RecommendedServices'] = df.apply(
        lambda row: row['RecommendedServices'] + ['FamilyPlan']
        if (row['Married'] == 'Yes' or row['Dependents'] == 'Yes')
        else row['RecommendedServices'],
        axis=1
    )

    # ------------------
    # 2️. 협업 필터링 추천 (User-based CF)
    # ------------------
    # 예시: CustomerId, Service, Rating 데이터 필요
    # 실제로는 고객 과거 서비스 이용/선호 데이터 필요
    service_ratings = df.melt(
        id_vars=['CustomerId'],
        value_vars=['OnlineSecurity', 'OnlineBackup', 'TechSupport', 'UnlimitedData'],
        var_name='Service',
        value_name='Used'
    )
    service_ratings['Rating'] = service_ratings['Used'].apply(lambda x: 1 if x == 'Yes' else 0)

    reader = Reader(rating_scale=(0,1))
    data = Dataset.load_from_df(service_ratings[['CustomerId','Service','Rating']], reader)

    # User-based CF
    trainset = data.build_full_trainset()

    # 0 벡터 고객 제거
    non_zero_users = service_ratings.groupby('CustomerId')['Rating'].sum()
    valid_users = non_zero_users[non_zero_users > 0].index.tolist()
    data_valid = service_ratings[service_ratings['CustomerId'].isin(valid_users)]
    trainset = Dataset.load_from_df(data_valid[['CustomerId','Service','Rating']], reader).build_full_trainset()

    sim_options = {'name': 'cosine', 'user_based': True}
    algo = KNNBasic(sim_options=sim_options)
    algo.fit(trainset)

    # 각 고객별 top-N 추천
    top_n = {}
    for uid in trainset.all_users():
        user_inner_id = uid
        user_raw_id = trainset.to_raw_uid(uid)
        items = trainset.all_items()
        predictions = [algo.predict(user_raw_id, trainset.to_raw_iid(iid)) for iid in items]
        predictions.sort(key=lambda x: x.est, reverse=True)
        top_n[user_raw_id] = [pred.iid for pred in predictions[:2]]  # 상위 2개 서비스 추천

    # 추천 합치기
    df['RecommendedServices'] = df.apply(
        lambda row: list(set(row['RecommendedServices'] + top_n.get(row['CustomerId'], []))),
        axis=1
    )

    return df



if __name__ == "__main__":
    df = pd.read_csv("data/processed/telco_cleaned_data.csv", encoding="utf-8")
    df = generate_recommendations(df)

    print(df[['CustomerId', 'cluster_name', 'RecommendedServices']].head())
    