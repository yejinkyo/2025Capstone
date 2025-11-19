import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import SVD
from surprise import get_dataset_dir 
import numpy as np
from sentence_transformers import SentenceTransformer, util
import joblib

#lr 이탈 예측 모델 호출
from train_model import train_and_evaluate

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


# ------------------
# 3. 대조분석 추천
# ------------------

#<사용자 입력 변환 함수>
#모델 입력 정답 스키마
MODEL_FEATURE_LIST = [
    'Gender', 'Age', 'AgeGroup', 'Married', 'Dependents', 'noDependents', 
    'Referrals', 'noReferrals', 'PaperlessBilling', 'PaymentMethod', 
    'OnlineSecurity', 'OnlineBackup', 'TechSupport', 'UnlimitedData', 
    'AvgDownloadGB', 'CustomerLTV', 'SatisScore', 'TotalExtraDataCharge', 
    'AvgRoamCharge', 'TotalRoamCharge', 'Tenure_month', 'Sum_charge', 
    'Monthly_charge', 'ServiceDuration', 'CLTV_monthly', 'TotalOtherCharges', 
    'LTVPerSatis', 'Is_Manual_Payment'
]

#값 매핑 사전
VALUE_MAPPING = {
    # 성별 통일
    '남자': '남성', '남': '남성', 'man': '남성', 'male': '남성',
    '여자': '여성', '여': '여성', 'woman': '여성', 'female': '여성',
    
    # 긍정/부정 통일
    '가입': 'Yes', '사용': 'Yes', '예': 'Yes', 'true': 'Yes', '1': 'Yes',
    '미가입': 'No', '미사용': 'No', '아니요': 'No', 'false': 'No', '0': 'No'
}

#이진 열
BINARY_COLS = [
    'Married', 'Dependents', 'Referrals', 'PaperlessBilling', 
    'OnlineSecurity', 'OnlineBackup', 'TechSupport', 'UnlimitedData'
]


def process_user_input_raw(consult_text: str, A_features_raw: dict) -> tuple[str, pd.DataFrame]:
    """
    사용자의 입력을 받아 학습 데이터와 동일한 형태(Raw Schema)의 DataFrame을 만듭니다.
    입력되지 않은 값은 0으로 채웁니다.
    """

    feature_data = {col: 0 for col in MODEL_FEATURE_LIST}

    # [기초 데이터 매핑] 사용자 입력값 채우기
    for raw_key, raw_value in A_features_raw.items():

        # 모델 피처에 없더라도, Age나 CustomerID는 처리
        if raw_key not in MODEL_FEATURE_LIST and raw_key not in ['Age', 'CustomerID']: 
            continue

        clean_val = str(raw_value).strip()

        # (A) 성별 처리
        if raw_key == 'Gender':
            feature_data['Gender'] = VALUE_MAPPING.get(clean_val, clean_val)

        # (B) Yes/No 처리
        elif raw_key in BINARY_COLS:
            mapped_val = VALUE_MAPPING.get(clean_val.lower(), clean_val)
            if mapped_val.lower() == 'yes': mapped_val = 'Yes'
            if mapped_val.lower() == 'no': mapped_val = 'No'
            feature_data[raw_key] = mapped_val

        # (C) Age -> AgeGroup 자동 생성
        elif raw_key == 'Age':
            try:
                age_val = float(clean_val)
                feature_data['Age'] = age_val
                # AgeGroup 자동 계산
                if 'AgeGroup' not in A_features_raw:
                    feature_data['AgeGroup'] = f"{(int(age_val) // 10) * 10}대"
            except: pass

        # (D) 나머지 (수치형 등)
        else:
            try:
                feature_data[raw_key] = float(clean_val)
            except ValueError:
                feature_data[raw_key] = clean_val # 문자열(PaymentMethod 등)은 그대로
    #[파생변수 자동 계산]

    # (1) TotalOtherCharges = ExtraData + Roam
    # (입력이 없으면 0 + 0 = 0이 됨)
    extra_data = feature_data.get('TotalExtraDataCharge', 0)
    roam_charge = feature_data.get('TotalRoamCharge', 0)
    feature_data['TotalOtherCharges'] = extra_data + roam_charge

    # (2) CLTV_monthly = CustomerLTV / ServiceDuration
    ltv = feature_data.get('CustomerLTV', 0)
    duration = feature_data.get('ServiceDuration', 0)
    
    if duration > 0: # 0으로 나누기 방지
        feature_data['CLTV_monthly'] = ltv / duration
    else:
        feature_data['CLTV_monthly'] = 0.0

    # (3) LTVPerSatis = CustomerLTV / SatisScore
    satis = feature_data.get('SatisScore', 0) 
    
    if satis > 0: # 0으로 나누기 방지
        feature_data['LTVPerSatis'] = ltv / satis
    else:
        feature_data['LTVPerSatis'] = 0.0 

    # (4) Is_Manual_Payment (수동 결제 여부)
    # PaymentMethod가 '신용카드'나 '이체/메일확인'이면 1, 아니면 0
    manual_payment_list = ['신용카드', '이체/메일확인']
    current_method = feature_data.get('PaymentMethod', '')
    
    if current_method in manual_payment_list:
        feature_data['Is_Manual_Payment'] = 1
    else:
        feature_data['Is_Manual_Payment'] = 0

    A_input_df = pd.DataFrame([feature_data], columns=MODEL_FEATURE_LIST)
    
    return consult_text, A_input_df

#<lr 이탈 예측 확률 함수>
def predict_churn_probability(model, A_input_df, feature_list):
    """
    로드된 모델(Pipeline 권장)과 사용자 A의 입력 데이터를 사용해 이탈 확률을 예측합니다.
    """
    # 1. 모델 존재 여부 확인
    if model is None:
        print("  - [알림] 모델 객체가 로드되지 않아 예측을 건너뜁니다. (확률 0.0 반환)")
        return 0.0
    
    # 2. 모델이 학습할 때 사용한 피처 순서대로 데이터 정렬
    # (만약 A_input_df에 없는 컬럼이 있다면 0으로 채워서 형태를 맞춰줍니다)
    try:
        # 빈 데이터프레임 생성 (학습 데이터 스키마 기준)
        A_features = pd.DataFrame(0, index=[0], columns=feature_list)
        
        # A_input_df의 값을 A_features에 덮어씌우기
        for col in A_input_df.columns:
            if col in feature_list:
                # 데이터 타입이 다르면 에러가 날 수 있으므로 값 복사 시 주의
                A_features[col] = A_input_df[col].values[0]

    except Exception as e:
        print(f"  - [예측 오류] 데이터 정렬 및 매핑 실패: {e}")
        return 0.0
    
    # 3. 이탈 확률(Class 1) 예측
    try:
        # predict_proba는 [[Class0_Prob, Class1_Prob]] 형태의 배열을 반환합니다.
        # [0][1]을 가져와야 '이탈(1)' 확률이 됩니다.
        churn_probability = model.predict_proba(A_features)[0][1]
        
    except ValueError as e:
        print(f"  - [치명적 오류] 모델 예측 실패: {e}")
        print("    👉 힌트: 입력 데이터에 문자열('남성', 'Yes' 등)이 포함되어 있습니다.")
        print("    👉 만약 모델이 Pipeline이 아니라면, 문자열을 숫자로 변환(OHE)하는 과정이 빠져있을 수 있습니다.")
        return 0.0
    except Exception as e:
        print(f"  - [예측 오류] 알 수 없는 에러: {e}")
        return 0.0

    return churn_probability

#<코사인 유사도 함수>
model_s = SentenceTransformer('jhgan/ko-sroberta-multitask')

#텔코 코퍼스 임베딩
TEXT_CORPUS_PATH = 'data/processed/telco_narrative_corpus.csv'

df_telco_text_raw = pd.read_csv(TEXT_CORPUS_PATH)
corpus_sentences = df_telco_text_raw['text'].fillna("").tolist()
corpus_embeddings = model_s.encode(corpus_sentences, normalize_embeddings=True)

def find_most_similar_customer_B(A_consult_text, model, corpus_embeddings, df_telco_text_raw):
    """
    A의 상담 텍스트와 가장 유사한 B(1명)를 S-BERT로 찾기
    """
    # 1. A의 임베딩 계산 (정규화)
    A_embedding = model.encode([A_consult_text], normalize_embeddings=True)

    # 2. 유사도 계산 (A vs 전체 코퍼스)    
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

#<유사 군집 분석>

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
        print(f"\n--- 오류: 유사 고객 (B) ID '{b_customer_id}'를 정형 데이터에서 찾을 수 없습니다.")
        return pd.DataFrame(columns=df.columns)
    
    # 컬럼명 확인 (Churn Label 또는 ChurnLabel)
    churn_col = 'Churn Label' if 'Churn Label' in df.columns else 'ChurnLabel'

    # 2. B와 동일 군집, 이탈X 고객 (C, D...) 찾기
    retained_neighbors_df = df[
        (df['kmeans_cluster_id'] == b_cluster_id) &
        (df['CustomerID'] != b_customer_id) &
        (df[churn_col] == 'No') # 이탈 안 한 고객 (문자열 'No' 기준)
    ]

    print(f"\n--- 6. 유사 고객 그룹 (C, D...) 탐색 ---")
    if retained_neighbors_df.empty:
        print(f"  - 클러스터 {b_cluster_id} 내에 비교할 만한 다른 이탈 방지 고객이 없습니다.")
    else:
        print(f"  - 총 {len(retained_neighbors_df)}명의 유사 고객 그룹(이탈X) 발견.")

    return retained_neighbors_df

#<대조분석 코드>

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
        return recommendations
    
    # (1) 이진형 서비스 분석 (가입률 Gap)
    # 계산을 위해 Yes/No를 1/0으로 변환
    map_dict = {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0}
    
    # 분석할 이진 컬럼 (A 데이터에 존재하는 것만)
    valid_bin_cols = [c for c in BINARY_COLS if c in A_input_df.columns]

    if valid_bin_cols:
        # A의 상태 (숫자로 변환)
        a_binary = A_input_df[valid_bin_cols].replace(map_dict).apply(pd.to_numeric, errors='coerce').fillna(0).iloc[0]
        
        # 그룹의 평균 상태 (가입률)
        neighbors_mean = retained_neighbors_df[valid_bin_cols].replace(map_dict).apply(pd.to_numeric, errors='coerce').fillna(0).mean()
        
        # 차이 계산 (그룹 가입률 - A 가입 상태)
        # 예: 그룹(0.8) - A(0) = 0.8 (추천 대상)
        gap = neighbors_mean - a_binary
        
        # 차이가 0.5 (50%) 이상인 서비스 필터링
        recommend_binary = gap[gap >= 0.5].sort_values(ascending=False)

        # 결과 저장
        for service, val in recommend_binary.items():
            rate = float(neighbors_mean[service])
            msg = f"✅ [{service}] 가입 추천 (유사그룹 가입률: {rate:.0%})"
            recommendations.append(msg)

    
    # (2) 범주형 서비스 분석 (최빈값 Mode)
    # ---------------------------------------
    cat_cols = ['PaymentMethod', 'Contract', 'Internet Service'] # 분석할 카테고리
    valid_cat_cols = [c for c in cat_cols if c in A_input_df.columns and c in retained_neighbors_df.columns]

    for col in valid_cat_cols:
        a_val = A_input_df[col].iloc[0] # A의 값
        
        # 그룹의 최빈값(Mode) 찾기
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
        print("  - 분석 결과: 고객 A는 이미 유사 그룹이 사용하는 주요 서비스를 대부분 이용 중입니다.")
        
        # (참고) A가 현재 가입 중인 서비스 출력
        if 'a_binary' in locals():
            a_subscribed = a_binary[a_binary == 1].index.tolist()
            if a_subscribed:
                print(f"  - (참고) 고객 A 가입 서비스: {', '.join(a_subscribed)}")
    else:
        print("  - 분석 결과: 유사 고객 그룹은 이용하지만 고객 A는 이용하지 않는 서비스 발견")
        print("  - 추천 서비스 목록:")
        for i, msg in enumerate(recommendations, 1):
            print(f"    {i}. {msg}")
        print("\n  => 위 서비스 가입을 추천합니다.")

    return recommendations

#<메인 통합 함수>
def generate_recommendations_contrast(user_consult_text, user_raw_features, 
                                      lr_model, feature_list,
                                      sbert_model, corpus_embeddings, 
                                      df_telco_text_raw, df_telco_clustering):
    """
    전체 대조 분석 추천 프로세스 실행
    """
    print("="*50)
    print("      고객 이탈 방지 추천 시스템")
    print("="*50)

    # --- 1. 사용자 A 입력 처리 ---
    A_consult_text, A_input_df = process_user_input_raw(user_consult_text, user_raw_features)
    print("--- 1. 사용자 A 입력 처리 완료 ---\n")
    print(f"  - 상담 텍스트: {A_consult_text}")

    # --- 2. 이탈 확률 예측 (LR Model) ---
    churn_prob = predict_churn_probability(lr_model, A_input_df, feature_list)
    print(f"\n--- 2. 사용자 A 이탈 확률 예측 (LR Model) ---")
    print(f"  - 예측된 이탈 확률: {churn_prob:.2%}")

    # --- 3. 가장 유사한 고객 B 찾기 (S-BERT) ---
    b_customer_id, b_similarity = find_most_similar_customer_B(
        A_consult_text,
        sbert_model,
        corpus_embeddings,
        df_telco_text_raw
    )

    if b_customer_id is None:
        return churn_prob, []
    
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

    try:
        TEXT_CORPUS_PATH = 'data/processed/telco_narrative_corpus.csv'
        CLUSTER_DATA_PATH = 'data/processed/telco_cleaned_data.csv'
        
        print("📂 데이터 파일 로딩...")
        df_telco_text_raw = pd.read_csv(TEXT_CORPUS_PATH)
        df_telco_clustering = pd.read_csv(CLUSTER_DATA_PATH)
        print("  - 데이터 로드 완료.")

        from train_model import train_and_evaluate
        print("🧠 모델 학습 및 객체 생성 중...")

        X_data = df_telco_clustering.copy()
        for col in MODEL_FEATURE_LIST:
            if col not in X_data.columns: X_data[col] = 0
        X_train_final = X_data[MODEL_FEATURE_LIST]

        # 타겟 변수 (Yes/No -> 1/0 변환)
        y_train_final = df_telco_clustering['Churn Label'].apply(lambda x: 1 if x == 'Yes' else 0)
        
        # ★ 학습 함수 호출하여 모델 객체 받기 (테스트용 split 없이 전체 데이터 학습 가정)
        lr_model, _ = train_and_evaluate(X_train_final, y_train_final, X_train_final, y_train_final)
        
        print("✅ 모델 준비 완료.")

        # [3] S-BERT 및 코퍼스 임베딩 생성
        print("⏳ 텍스트 코퍼스 임베딩 생성 중...")
        sbert_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
        corpus_sentences = df_telco_text_raw['text'].fillna("").tolist()
        corpus_embeddings = sbert_model.encode(corpus_sentences, normalize_embeddings=True)
        print("✅ 임베딩 완료.\n")

        user_consult_text = "요금제가 너무 비싸서 부담스러워요."
        
        user_raw_features = {
            'CustomerID': 'NewUser_A',
            'Gender': '남자',            # -> 남성
            'Age': 30,                   # -> AgeGroup: 30대
            'PaymentMethod': '계좌이체',  # -> 추천: 신용카드
            'Streaming TV': '미가입',     # -> 추천: 가입 (Yes)
            'OnlineSecurity': '미가입',   # -> 추천: 가입 (Yes)
            'CustomerLTV': 5000,
            'ServiceDuration': 10
        }

        generate_recommendations_contrast(
            user_consult_text,
            user_raw_features,
            lr_model,           # 위에서 학습된 모델 객체 전달
            MODEL_FEATURE_LIST,
            sbert_model,
            corpus_embeddings,
            df_telco_text_raw,
            df_telco_clustering
        )

    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")
        # (디버깅용) 에러 상세 정보 출력
        import traceback
        traceback.print_exc()





