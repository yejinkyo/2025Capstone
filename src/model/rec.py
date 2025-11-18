import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import SVD
from surprise import get_dataset_dir 
import numpy as np
from sentence_transformers import SentenceTransformer, util

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


#lr 이탈 예측 모델 함수
def predict_churn_probability(model, A_input_df, feature_list):
    """
    로드된 모델과 사용자 A의 입력 데이터를 사용해 이탈 확률을 예측합니다.
    """
    if model is None:
        print("  - 모델이 로드되지 않아 예측을 건너뜁니다.")
        return 0.0 # 모델이 없을 경우 0% 반환

    # 1. 모델이 학습한 피처 순서대로 데이터 정렬
    try:
        A_features = A_input_df[feature_list]
    except KeyError as e:
        print(f"  - 예측 오류: 입력 데이터에 필요한 칼럼이 없습니다 - {e}")
        return 0.0

    # 2. 이탈 확률(1) 예측
    # predict_proba는 [class_0_prob, class_1_prob] 반환
    churn_probability = model.predict_proba(A_features)[0][1]

    return churn_probability

#코사인 유사도 함수
model_s = SentenceTransformer('jhgan/ko-sroberta-multitask')

def find_most_similar_customer_B(A_consult_text, model, corpus_embeddings, df_telco_text_raw):
    """
    A의 상담 텍스트와 가장 유사한 B(1명)를 S-BERT로 찾기
    """
    # 1. A의 임베딩 계산 (정규화)
    A_embedding = model.encode([A_consult_text], normalize_embeddings=True)

    # 2. 유사도 계산 (A vs 전체 코퍼스)
    # util.cos_sim은 이미 정규화된 벡터를 사용하므로 빠릅니다.

    churn_df = df[['CustomerId', 'ChurnLabel', 'ChurnReason']]
    corpus_sentences = churn_df.tolist()
    corpus_embeddings = model_s.encode(corpus_sentences,
                                 normalize_embeddings=True)
    
    sim_scores = util.cos_sim(A_embedding, corpus_embeddings)[0].numpy()

    # 3. Top-1 (User B) 인덱스 찾기
    b_index = np.argmax(sim_scores)
    b_similarity = float(sim_scores[b_index])

    # 4. User B의 CustomerID 및 텍스트 찾기
    b_customer_row = df_telco_text_raw.iloc[b_index]
    b_customer_id = b_customer_row['CustomerID']

    print(f"\n--- 4. 가장 유사한 고객 (B) 탐색 완료 (S-BERT) ---")
    print(f"  - 대상 고객 ID (B): {b_customer_id}")
    print(f"  - 텍스트 유사도: {b_similarity:.4f}")

    return b_customer_id, b_similarity

#유사 군집 분석
def find_retained_neighbors(b_customer_id, df):
    """
    B의 군집을 찾고, 군집 내 이탈X 고객(C,D)의 DataFrame을 반환합니다.
    """

    # 1. B의 군집 ID 찾기 (df_telco_clustering 사용)
    try:
        b_row = df[df['CustomerID'] == b_customer_id].iloc[0]
        b_cluster_id = b_row['kmeans_cluster_id']
        print(f"\n--- 5. 유사 고객 (B)의 군집 탐색 ---")
        print(f"  - B의 소속 군집: {b_cluster_id}")
    except IndexError:
        print(f"\n--- 오류: 유사 고객 (B) ID '{b_customer_id}'를 df(정형 데이터)에서 찾을 수 없습니다.")
        # 빈 DataFrame 반환
        return pd.DataFrame(columns=df.columns)

    # 2. B와 동일 군집, 이탈X 고객 (C, D...) 찾기
    retained_neighbors_df = df[
        (df['kmeans_cluster_id'] == b_cluster_id) &
        (df['CustomerID'] != b_customer_id) &
        (df['Churn Label'] == 'No') # 이탈 안 한 고객
    ]

    print(f"\n--- 6. 유사 고객 그룹 (C, D...) 탐색 ---")
    if retained_neighbors_df.empty:
        print(f"  - 클러스터 {b_cluster_id} 내에 비교할 만한 다른 이탈 방지 고객이 없습니다.")
    else:
        print(f"  - 총 {len(retained_neighbors_df)}명의 유사 고객 그룹(이탈X) 발견.")

    return retained_neighbors_df

def perform_contrastive_analysis(A_input_df, retained_neighbors_df):
    """
    A의 정보와 유사 그룹(C,D)의 정보를 대조 분석하여 서비스를 추천합니다.
    1. 이진형 서비스 (가입 안 한 것 추천)
    2. 범주형 서비스 (대세인 옵션으로 변경 추천)
    """
    print(f"\n--- 7. 이탈 방지 대책 추천 (대조 분석) ---")

    recommendations = []

    # 1. C, D... 그룹이 비어있는지 확인
    if retained_neighbors_df.empty:
        print("  - 비교할 유사 그룹이 없어 추천을 생성할 수 없습니다.")
        return pd.Series(dtype=float) # 빈 Series 반환

    # 2. 대조 분석 수행
    # 비교할 서비스 피처 목록 (OHE 기준)
    #이진 피쳐
    binary_features = [
        'Married', 'Dependents', 'Referrals', 'PaperlessBilling', 
        'OnlineSecurity', 'TechSupport', 'OnlineBackup', 'UnlimitedData'
    ]
    #카테고리 피쳐
    categorical_features = ['PaymentMethod']

    # 매핑 사전
    map_dict = {'Yes': 1, 'No': 0}

    # 데이터프레임에 존재하는 컬럼만 필터링
    valid_bin_cols = [c for c in binary_features if c in A_input_df.columns]
    
    # 숫자(1/0)로 변환
    a_binary = A_input_df[valid_bin_cols].replace(map_dict).iloc[0]
    neighbors_mean = retained_neighbors_df[valid_bin_cols].replace(map_dict).mean()
    
    # 차이 계산 (그룹 평균 - 내 상태)
    gap = neighbors_mean - a_binary
    
    # 차이가 0.5 (50%) 이상인 서비스 필터링
    recommend_binary = gap[gap >= 0.5].sort_values(ascending=False)

    # 추천 리스트에 추가
    for service, score in recommend_binary.items():
        service_name = service.replace('_', ' ')
        # 요청하신 포맷에 맞춘 메시지 생성
        msg = f"✅ [{service_name}] 가입 추천 (유사그룹 가입률: {neighbors_mean[service]:.0%})"
        recommendations.append(msg)

    #---------
    #카테고리 피쳐
    #---------
    categorical_features = ['PaymentMethod']

    valid_cat_cols = [c for c in categorical_features if c in A_input_df.columns]

    for col in valid_cat_cols:
        a_val = A_input_df[col].iloc[0] # A의 값
        
        if not retained_neighbors_df.empty:
            # 그룹의 최빈값
            mode_res = retained_neighbors_df[col].mode()
            if not mode_res.empty:
                group_mode = mode_res[0]
                # 최빈값의 비율 계산
                group_ratio = retained_neighbors_df[col].value_counts(normalize=True)[group_mode]
                
                # A와 다르고, 그룹의 40% 이상이 사용할 때 추천
                if a_val != group_mode and group_ratio >= 0.4:
                    msg = f"🔄 [{col}] 변경 추천: '{a_val}' ➔ '{group_mode}' (유사그룹 {group_ratio:.0%} 이용)"
                    recommendations.append(msg)

    # 3. 결과 출력
    if not recommendations:
        # 추천할 게 없을 때
        print("  - 분석 결과: 고객 A는 이미 유사 그룹이 사용하는 주요 서비스를 대부분 이용 중입니다.")
        
        # (참고) A가 현재 가입 중인 서비스 출력 로직
        # 1인 서비스들만 추출
        a_subscribed = a_binary[a_binary == 1].index.tolist()
        if a_subscribed:
            clean_names = [s.replace('_', ' ') for s in a_subscribed]
            print(f"  - (참고) 고객 A 현재 가입 서비스: {', '.join(clean_names)}")
            
    else:
        # 추천할 게 있을 때
        print("  - 분석 결과: 유사 고객 그룹은 이용하지만 고객 A는 이용하지 않는 서비스(또는 다른 패턴) 발견")
        print("  - 추천 서비스 목록:")
        
        # 리스트에 담긴 메시지 출력
        for i, msg in enumerate(recommendations, 1):
            print(f"    {i}. {msg}")
            
        print("\n  => 위 서비스 가입 및 변경을 추천합니다.")

    return recommendations

def generate_recommendations_contrast(user_consult_text, user_raw_features, 
                                      lr_model, feature_list,
                                      sbert_model, corpus_embeddings, 
                                      df_telco_text_raw, df_telco_clustering):
    """
    대조 분석 기반 추천
    """
    A_input_df = pd.DataFrame([user_raw_features])
    A_consult_text = ""
    print("--- 1. 사용자 A 입력 처리 완료 ---\n")
    print(f"  - 상담 텍스트: {A_consult_text}")

    # --- 2. 이탈 확률 예측 (Hist Model) ---
    # (새로 추가된 부분)
    churn_prob = predict_churn_probability(lr_model, A_input_df, feature_list)
    print(f"\n--- 2. 사용자 A 이탈 확률 예측 (Hist) ---")
    print(f"  - 예측된 이탈 확률: {churn_prob:.2%}")

    # --- 3. 가장 유사한 고객 B 찾기 (S-BERT) ---
    b_customer_id, b_similarity = find_most_similar_customer_B(
        A_consult_text,
        model_s,
        corpus_embeddings,
        df_telco_text_raw
    )

    # --- 4. B와 같은 군집의 C, D... 찾기 ---
    retained_neighbors_df = find_retained_neighbors(
        b_customer_id,
        df_telco_clustering
    )

    # --- 5. A와 (C, D...) 대조 분석 및 추천 ---
    recommendations = perform_contrastive_analysis(
        A_input_df,
        retained_neighbors_df
    )

    print("\n--- 8. 추천 프로세스 완료 ---")

    return churn_prob, recommendations




if __name__ == "__main__":
    df = pd.read_csv("data/processed/telco_cleaned_data.csv", encoding="utf-8")
    df = generate_recommendations(df)

    print(df[['CustomerId', 'cluster_name', 'RecommendedServices']].head())