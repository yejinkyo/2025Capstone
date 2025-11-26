import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import SVD
from surprise import get_dataset_dir 
import numpy as np
from sentence_transformers import SentenceTransformer, util
import joblib
import os

# ==========================================
# 0. 사전 준비 단계 (서버 시작 시 1회 실행 가정)
# ==========================================
def prepare_data_pipeline(df):
    """
    데이터 전처리 및 통계 계산, 모델 학습
    """
    # 인기 서비스 집계 
    service_candidates = ['OnlineSecurity', 'OnlineBackup', 'TechSupport', 'UnlimitedData']
    available_cols = [col for col in service_candidates if col in df.columns]

    popular_services = []
    if available_cols:
        counts = {}
        for col in available_cols:
            # 데이터 값이 'Yes'인 경우만 카운트
            counts[col] = df[col].value_counts().get('Yes', 0)
        popular_services = sorted(counts, key=counts.get, reverse=True)[:2]
    
    # 룰 기반 추천용 통계 계산
    stats = {
        'median_download_gb': df['AvgDownloadGB'].median(),
        'median_satis_score': df['SatisScore'].median(),
        'avg_churn_score': df['ChurnScore'].mean(),
        'popular_services': popular_services,
        'cluster_map': {
            '표준 단기 고객 (월정액)': ['Standard Plan'],
            '알뜰형 장기 고객 (2년 약정, 저CLTV)': ['Budget Plan'],
            '기술선호형 중장기 고객 (월정액)': ['Tech Plan'],
            '초단기 신규 고객 (최대 이탈 위험군)': ['Trial Plan'],
            '표준 중기 고객 (1년 약정)': ['Standard Plan'],
            '고가치 장기 고객 (월정액, 고CLTV)': ['Premium Plan'],
        }
    }

    # 협업 필터링 모델 학습
    service_ratings = df.melt(
        id_vars=['CustomerId'],
        value_vars=available_cols,
        var_name='Service',
        value_name='Used'
    )
    service_ratings['Rating'] = service_ratings['Used'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    non_zero_users = service_ratings.groupby('CustomerId')['Rating'].sum()
    valid_users = non_zero_users[non_zero_users > 0].index.tolist()
    data_valid = service_ratings[service_ratings['CustomerId'].isin(valid_users)]
    
    return df, stats

def create_service_usage_cols(df, service_cols, threshold=0):
    """
    서비스 사용 여부를 0/1로 변환하는 함수
    threshold: 이 값보다 크면 1로 간주
    """
    df_bin = df.copy()
    for col in service_cols:
        df_bin[col] = (df[col] > threshold).astype(int)
    return df_bin

# ==========================================
# 1. 룰 기반 추천 함수 (User Input)
# ==========================================

def rule_based_recommendations(user_row, stats):
    """
    룰 기반 추천 생성하는 함수

    Parameters:
        user_row: pd.Series, 단일 고객 데이터 행
        stats: dict, 전체 유저 기준 통계값

    Returns:
        recommendations: list, 추천 서비스 목록
    """
    recommendations = []

    # 1. 군집 기반 추천
    cluster = user_row.get('cluster_name')
    recommendations.extend(stats['cluster_map'].get(cluster, []))

    # 2. 사용량 기반 (중앙값 비교)
    if user_row['AvgDownloadGB'] > stats['median_download_gb']:
        recommendations.append('UnlimitedData')
    
    # 3. 만족도 기반 (중앙값 비교)
    if user_row['SatisScore'] < stats['median_satis_score']:
        recommendations.append('TechSupport')

    # 4. 이탈 위험 기반 (평균값 비교)
    if user_row['ChurnScore'] > stats['avg_churn_score']:
        recommendations.append('OnlineBackup')

    # 5. 디지털 선호 기반
    digital_payment_methods = ['이체/메일확인', '신용카드', 'Electronic check']
    if user_row['PaperlessBilling'] == 'Yes' and user_row['PaymentMethod'] in digital_payment_methods:
        recommendations.append('OnlineSecurity')

    # 6. 가족 기반
    pass

    return list(set(recommendations)) # 중복 제거 

# ==========================================
# 2. 협업 필터링 추천 함수
# ==========================================
def convert_yes_no(df, service_cols):
    df_bin = df.copy()
    for col in service_cols:
        df_bin[col] = df_bin[col].map({"Yes": 1, "No": 0})
        df_bin[col] = df_bin[col].fillna(0) 
    return df_bin

def build_user_item_matrix(df, service_cols):
    """
    사용자-아이템 행렬 생성 함수
    """
    records = []

    for idx, row in df.iterrows():
        user = str(row["CustomerId"])

        for service in service_cols:
            val = row[service]

            # --- (1) Yes/No 처리 ---
            if isinstance(val, str):
                if val.lower() == "yes":
                    rating = 1
                elif val.lower() == "no":
                    rating = 0
                else:
                    # 숫자로 바꿀 수 있을 때만 변환
                    try:
                        rating = int(float(val))
                    except:
                        rating = 0
            else:
                # 숫자형 (float/int)
                try:
                    rating = int(val)
                except:
                    rating = 0

            # rating이 1 이상일 때만 "사용"으로 간주
            if rating > 0:
                records.append([user, service, 1])

    ratings_df = pd.DataFrame(records, columns=['user', 'item', 'rating'])
    return ratings_df


def train_cf_model(ratings_df):
    """
    ratings_df: user-item-rating 데이터프레임
    return: 학습된 algo, trainset
    """
    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(ratings_df, reader)

    trainset, testset = train_test_split(data, test_size=0.2)

    algo = SVD(n_factors=50, n_epochs=20, biased=False)
    algo.fit(trainset)

    return algo, trainset

def cf_recommendations(user_id, algo, trainset, service_cols=None, top_n=10):
    """
    특정 user_id에 대해 협업 필터링 기반 추천 생성

    Parameters:
        user_id: str, 고객 ID
        algo: 학습된 협업 필터링 모델
        trainset: 학습 데이터셋
        top_n: int, 추천할 서비스 개수

    Returns:
        recommendations: list, 추천 서비스 목록
    """
    user_id = str(user_id)

    try:
        # 문자열 ID를 내부 정수 ID로 변환
        inner_uid = trainset.to_inner_uid(user_id)
    except ValueError:
        # 모델에 없는 신규 유저는 빈 리스트 반환
        return []
    
    # 이미 사용 중인 아이템들의 내부 ID(iid)를 Set으로 저장)
    used_items = set(trainset.to_raw_iid(iid) for (iid, _) in trainset.ur[inner_uid])

    predictions = []
    for service in service_cols:
        if service in used_items:
            continue

        pred = algo.predict(user_id, service)
        predictions.append((service, pred.est))

    predictions.sort(key=lambda x: x[1], reverse=True)
    return [item for item, score in predictions[:top_n]]


# ------------------
# 3. 대조분석 추천
# ------------------

MODEL_FEATURE_LIST = [
    # [1] 기본 수치/이진형 (Raw Data 컬럼명 유지)
    'Gender', 'Age', 'Married', 'Dependents', 'noDependents', 
    'Referrals', 'noReferrals', 'PaperlessBilling', 
    'OnlineSecurity', 'OnlineBackup', 'TechSupport', 'UnlimitedData', 
    'AvgDownloadGB', 'CustomerLTV', 'SatisScore', 
    'TotalExtraDataCharge', 'AvgRoamCharge', 'TotalRoamCharge', 
    'Tenure_month', 'Sum_charge', 'Monthly_charge', 'ServiceDuration', 
    
    # [2] 파생변수 (자동 계산 대상)
    'CLTV_monthly', 'TotalOtherCharges', 'LTVPerSatis', 'Is_Manual_Payment',
    
    # [3] PaymentMethod OHE (계좌이체는 Baseline -> 둘 다 0일 때)
    'PaymentMethod_신용카드', 'PaymentMethod_이체/메일확인',
    
    # [4] AgeGroup OHE (20대는 Baseline -> 전부 0일 때)
    'AgeGroup_30대', 'AgeGroup_40대', 'AgeGroup_50대', 
    'AgeGroup_60대', 'AgeGroup_70대', 'AgeGroup_80대'
]

# 값 통일을 위한 매핑 사전 (사용자 입력 -> 학습 데이터 용어)
VALUE_MAPPING = {
    '남자': '남성', '남': '남성', 'male': '남성', 
    '여자': '여성', '여': '여성', 'female': '여성',
    '가입': 'Yes', '사용': 'Yes', '예': 'Yes', 'true': 'Yes', '1': 'Yes',
    '미가입': 'No', '미사용': 'No', '아니요': 'No', 'false': 'No', '0': 'No',
    # 결제수단 표준화
    '신용카드': '신용카드',
    '계좌이체': '계좌이체',
    '이체': '이체/메일확인', '메일확인': '이체/메일확인', '이체/메일확인': '이체/메일확인'
}

# 0/1이 아닌 Yes/No로 변환해야 할 이진형 컬럼들
BINARY_COLS = ['Married', 'PaperlessBilling', 'OnlineSecurity', 
               'OnlineBackup', 'TechSupport', 'UnlimitedData']



# 2. 사용자 입력 변환 함수
def process_user_input_raw(consult_text: str, A_features_raw: dict) -> tuple[str, pd.DataFrame]:
    """
    사용자 입력을 받아 모델 학습 데이터와 동일한 형태의 DataFrame을 만듭니다.
    - 20대 입력 시 -> AgeGroup 관련 컬럼 모두 0 (Baseline)
    - 계좌이체 입력 시 -> PaymentMethod 관련 컬럼 모두 0 (Baseline)
    """

    # 1. 모든 컬럼을 0으로 초기화
    feature_data = {col: 0 for col in MODEL_FEATURE_LIST}

    # 2. 입력 데이터 매핑 및 처리
    for raw_key, raw_value in A_features_raw.items():
        clean_val = str(raw_value).strip()

        # (A) 성별 처리 (학습 데이터에선 0/1로 변환됨)
        if raw_key == 'Gender':
            mapped = VALUE_MAPPING.get(clean_val, clean_val)
            # 남성이면 1, 여성이면 0
            feature_data['Gender'] = 1 if mapped in ['남성', 'Male'] else 0
            
        # (B) 이진형 처리 (Yes -> 1, No -> 0)
        elif raw_key in BINARY_COLS:
            mapped = VALUE_MAPPING.get(clean_val.lower(), clean_val)
            feature_data[raw_key] = 1 if mapped.lower() in ['yes', '1'] else 0

        # (C) Dependents & Referrals (대칭 변수 처리)
        elif raw_key == 'Dependents':
            mapped = VALUE_MAPPING.get(clean_val.lower(), clean_val)
            is_yes = 1 if mapped.lower() in ['yes', '1'] else 0
            feature_data['Dependents'] = is_yes
            feature_data['noDependents'] = 1 - is_yes # 반대값 설정

        # (D) 수치형 데이터 처리 (Age 등)
        elif raw_key in feature_data:
            try: feature_data[raw_key] = float(clean_val)
            except: pass

        # (E) Age -> AgeGroup OHE (핵심 로직!)
        if raw_key == 'Age':
            try:
                age = int(float(clean_val))
                feature_data['Age'] = age
                
                # 10단위로 끊어서 해당 그룹 찾기
                decade = (age // 10) * 10
                
                # 30대 이상일 때만 해당 컬럼을 1로 설정
                # 20대 이하는 아무것도 1로 설정하지 않으므로 All 0 (Baseline)이 됨
                if decade >= 30: 
                    target_col = f"AgeGroup_{decade}대"
                    if target_col in feature_data:
                        feature_data[target_col] = 1
            except: pass

        # (F) PaymentMethod OHE (핵심 로직!)
        if raw_key == 'PaymentMethod':
            mapped_pm = VALUE_MAPPING.get(clean_val, clean_val)
            
            # '신용카드'나 '이체/메일확인'일 때만 해당 컬럼 1로 설정
            # '계좌이체'면 아무것도 설정하지 않으므로 All 0 (Baseline)이 됨
            target_col = f"PaymentMethod_{mapped_pm}"
            if target_col in feature_data:
                feature_data[target_col] = 1

    # [3] 파생변수 자동 계산
    feature_data['TotalOtherCharges'] = feature_data.get('TotalExtraDataCharge', 0) + feature_data.get('TotalRoamCharge', 0)
    
    ltv = feature_data.get('CustomerLTV', 0)
    dur = feature_data.get('ServiceDuration', 1) # 0 나누기 방지
    satis = feature_data.get('SatisScore', 1)
    if dur == 0: dur = 1
    if satis == 0: satis = 1

    feature_data['CLTV_monthly'] = ltv / dur
    feature_data['LTVPerSatis'] = ltv / satis
    
    # Is_Manual_Payment (신용카드, 이체/메일확인이 1이면 수동)
    is_manual = 0
    if feature_data.get('PaymentMethod_신용카드') == 1 or feature_data.get('PaymentMethod_이체/메일확인') == 1:
        is_manual = 1
    feature_data['Is_Manual_Payment'] = is_manual

    # DataFrame으로 변환하여 반환
    return consult_text, pd.DataFrame([feature_data], columns=MODEL_FEATURE_LIST)


# 3. 이탈 확률 예측 함수 (불러온 모델 사용)
def predict_churn_probability(model, A_input_df):
    """
    변환된 데이터(A_input_df)를 모델에 넣어 이탈 확률을 예측합니다.
    """
    if model is None: return 0.0
    try:
        # A_input_df는 이미 MODEL_FEATURE_LIST 순서대로 만들어져 있음
        return model.predict_proba(A_input_df)[0][1]
    except Exception as e:
        print(f"  - [예측 오류] {e}")
        return 0.0
    
# 4. 코사인 유사도 계산 함수
def find_most_similar_customer_B(A_consult_text, model, corpus_embeddings, df_telco_text_raw):
    """
    A의 상담 텍스트와 가장 유사한 B(1명)를 S-BERT로 찾습니다.
    """
    # 1. 사용자 A 텍스트 임베딩
    A_embedding = model.encode([A_consult_text], normalize_embeddings=True)
    
    # 2. 코사인 유사도 계산
    sim_scores = util.cos_sim(A_embedding, corpus_embeddings)[0].numpy()
    
    # 3. 가장 높은 점수 찾기
    b_index = np.argmax(sim_scores)
    b_similarity = float(sim_scores[b_index])

    # 4. 해당 인덱스의 고객 ID 가져오기 (컬럼명 대소문자 처리)
    b_customer_row = df_telco_text_raw.iloc[b_index]
    b_id = None
    for name in ['CustomerID', 'CustomerId', 'customerID', 'id']:
        if name in b_customer_row: b_id = b_customer_row[name]; break
    if b_id is None: b_id = b_customer_row.iloc[0]

    print(f"\n--- 4. 가장 유사한 고객 (B) 탐색 완료 ---")
    print(f"  - ID: {b_id} (유사도: {b_similarity:.4f})")
    return b_id, b_similarity

# 5. 유사 군집 및 대조 분석 함수
def find_retained_neighbors(b_id, df):
    """B와 같은 군집에 있는 비이탈 고객들을 찾습니다."""
    id_col = 'CustomerID'
    for col in df.columns:
        if col.lower() in ['customerid', 'id']: id_col = col; break
    
    try:
        b_row = df[df[id_col] == b_id].iloc[0]
        b_cluster = b_row['kmeans_cluster_id']
        print(f"\n--- 5. 유사 고객 (B)의 군집 탐색: {b_cluster} ---")
    except:
        return pd.DataFrame()

    churn_col = 'Churn Label' if 'Churn Label' in df.columns else 'ChurnLabel'
    # 이탈하지 않은('No') 고객만 필터링
    neighbors = df[(df['kmeans_cluster_id'] == b_cluster) & (df[id_col] != b_id) & (df[churn_col] == 'No')]
    print(f"--- 6. 유사 군집 내 롤모델 {len(neighbors)}명 탐색 ---")
    return neighbors

def perform_contrastive_analysis(A_input_df, retained_neighbors_df):
    """
    대조 분석 및 추천
    """
    print(f"\n--- 7. 이탈 방지 대책 추천 (대조 분석) ---")
    recommendations = []

    if retained_neighbors_df.empty:
        print("  - [알림] 비교할 롤모델(유사 비이탈 그룹)이 없어 추천을 생략합니다.")
        return recommendations

    # ---------------------------------------
    # 1. 이진형 서비스 분석 (가입 여부)
    # ---------------------------------------
    # 비교할 이진 컬럼들
    bin_cols = ['Married', 'Dependents', 'Referrals', 'PaperlessBilling', 
                'OnlineSecurity', 'OnlineBackup', 'TechSupport', 'UnlimitedData']
    
    # (1) 나의 상태 (0 또는 1)
    # A_input_df는 이미 숫자(0/1)로 변환되어 있으므로 해당 컬럼만 뽑습니다.
    # .iloc[0]을 써서 Series가 아닌 'Series 안의 값'을 가져오도록 합니다. (중요!)
    my_status = {}
    for col in bin_cols:
        if col in A_input_df.columns:
            my_status[col] = A_input_df[col].iloc[0]
        elif f"{col}_Yes" in A_input_df.columns: # OHE된 이름일 경우
             my_status[col] = A_input_df[f"{col}_Yes"].iloc[0]

    # (2) 이웃들의 상태 (평균 가입률 계산)
    # 문자열('Yes'/'No')을 숫자로 변환 후 평균 계산
    map_dict = {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0}
    neighbors_mean = retained_neighbors_df[bin_cols].replace(map_dict).infer_objects(copy=False).mean()

    # (3) 차이 비교 및 추천 (이웃은 많이 쓰는데(>50%), 나는 안 씀(0))
    for col, val in my_status.items():
        group_rate = neighbors_mean.get(col, 0)
        
        # 나는 안 씀(0) AND 그룹 가입률이 50% 이상임
        if val == 0 and group_rate >= 0.5:
            recommendations.append(f"✅ [{col}] 가입 추천 (유사그룹 가입률: {group_rate:.0%})")


    # ---------------------------------------
    # 2. 범주형 서비스 분석 (결제 수단 등)
    # ---------------------------------------
    cat_cols = ['PaymentMethod', 'Contract']
    
    for col in cat_cols:
        if col in retained_neighbors_df.columns:
            # 그룹의 최빈값(Mode) 찾기
            n_mode = retained_neighbors_df[col].mode()
            if not n_mode.empty:
                mode_val = n_mode[0] # 예: '신용카드'
                ratio = retained_neighbors_df[col].value_counts(normalize=True)[mode_val]
                
                # 내가 그 방식을 안 쓰고 있는지 확인 (간단 로직)
                # 내 데이터의 OHE 컬럼(예: PaymentMethod_신용카드)이 1인지 확인
                
                is_using = False
                # 매핑된 OHE 컬럼명 만들기 (예: PaymentMethod_신용카드)
                target_ohe = f"{col}_{mode_val}"
                if target_ohe == 'PaymentMethod_계좌이체': # 계좌이체는 OHE 컬럼이 없음 (Baseline)
                    # 다른거 다 0이면 계좌이체임
                     if A_input_df.get('PaymentMethod_신용카드', 0).iloc[0] == 0 and \
                        A_input_df.get('PaymentMethod_이체/메일확인', 0).iloc[0] == 0:
                        is_using = True
                elif target_ohe in A_input_df.columns:
                    if A_input_df[target_ohe].iloc[0] == 1:
                        is_using = True
                
                # 이체/메일확인 예외 처리
                if mode_val in ['이체', '메일확인', '이체/메일확인']:
                     if A_input_df.get('PaymentMethod_이체/메일확인', 0).iloc[0] == 1:
                        is_using = True

                # 추천 조건: 내가 안 쓰고 있고, 그룹의 40% 이상이 쓸 때
                if not is_using and ratio >= 0.4:
                    recommendations.append(f"🔄 [{col}] 변경 추천: '{mode_val}' (유사그룹 {ratio:.0%} 이용)")


    # ---------------------------------------
    # 3. 결과 출력
    # ---------------------------------------
    if not recommendations:
        print("  - 분석 결과: 고객 A는 이미 유사 그룹이 사용하는 주요 서비스를 대부분 이용 중입니다.")
    else:
        print("  - 분석 결과: 유사 고객 그룹은 이용하지만 고객 A는 이용하지 않는 서비스 발견")
        print("  - 추천 서비스 목록:")
        for i, msg in enumerate(recommendations, 1):
            # 메시지에서 불필요한 기호 제거하고 깔끔하게 출력
            clean_msg = msg.replace("✅ ", "").replace("🔄 ", "")
            print(f"    {i}. {clean_msg}")
            
        print("\n  => 위 서비스 가입 및 변경을 추천합니다.")

    return recommendations

def generate_recommendations_contrast(user_text, user_feats, lr_model, sbert_model, corpus_emb, df_text, df_cluster):
    # 1. 입력 처리
    processed_text, A_df = process_user_input_raw(user_text, user_feats)
    print(f"--- 1. 입력 처리 완료: {processed_text}")
    
    # 2. 예측
    churn_prob = predict_churn_probability(lr_model, A_df)
    print(f"--- 2. 이탈 예측 확률: {churn_prob:.2%}")

    # 3. 유사도 & 대조
    b_id, _ = find_most_similar_customer_B(processed_text, sbert_model, corpus_emb, df_text)
    if b_id:
        neighbors = find_retained_neighbors(b_id, df_cluster)
        perform_contrastive_analysis(A_df, neighbors)
    
    return churn_prob

def load_resources():
    print("📦 리소스 로딩 중...")
    # 경로 설정 (환경에 맞게 수정하세요)
    LR_PATH = 'data/processed/lr_model.joblib'
    TEXT_PATH = 'data/processed/telco_narrative_corpus.csv'
    CLUSTER_PATH = 'data/processed/telco_cleaned_data.csv'
    EMBED_PATH = 'data/processed/corpus_embeddings.joblib'

    # 1. 모델 로드
    if os.path.exists(LR_PATH):
        lr_model = joblib.load(LR_PATH)
    else:
        print("❌ [오류] 모델 파일(lr_model.joblib)이 없습니다!")
        return None, None, None, None, None

    # 2. 데이터 로드
    df_text = pd.read_csv(TEXT_PATH)
    df_cluster = pd.read_csv(CLUSTER_PATH)
    sbert = SentenceTransformer('jhgan/ko-sroberta-multitask')

    if os.path.exists(EMBED_PATH):
        print("  - 저장된 임베딩 로드 중...")
        corpus_emb = joblib.load(EMBED_PATH)
    else:
        print("  - ⚠️ 저장된 임베딩 파일이 없습니다! (make_embedding.py 먼저 실행 권장)")
        print("  - (비상) 즉석 생성 중...")
        corpus_emb = sbert.encode(df_text['text'].fillna("").tolist(), normalize_embeddings=True)
    
    print("✅ 로드 완료.")
    return lr_model, sbert, corpus_emb, df_text, df_cluster


if __name__ == "__main__":

    df = pd.read_csv("data/processed/telco_cleaned_data.csv", encoding="utf-8")
    
    # 전역 통계값 및 모델 학습
    df, stats = prepare_data_pipeline(df)

    user = df.iloc[0] 
    print(user)

    # ========룰 기반 추천=========
    recs = rule_based_recommendations(user, stats)
    print("룰 기반 추천 서비스:", recs)

    # ========협업 필터링 추천=========
    service_cols = ['PaperlessBilling', 'OnlineSecurity', 'OnlineBackup', 'TechSupport', 'UnlimitedData']
    df_bin = convert_yes_no(df, service_cols)

    ratings_df = build_user_item_matrix(df, service_cols)
    print(ratings_df.head())

    algo, trainset = train_cf_model(ratings_df)

    user_id = df_bin.iloc[0]['CustomerId']

    # recs = cf_recommendations(
    #     user_id=user_id,
    #     algo=algo,
    #     trainset=trainset,
    #     service_cols=service_cols,
    #     top_n=5
    # )

    # print(recs)

    # ========대조분석 추천=========

    try:
        # 1. 리소스 로드 (한 번만 수행)
        lr_model, sbert_model, corpus_embeddings, df_text, df_cluster = load_resources()

        if lr_model is not None:
            # 2. 사용자 입력 시나리오 (테스트)
            user_text_input = "요금제가 너무 비싸서 부담스러워요."
            user_features_input = {
                'CustomerID': 'NewUser',
                'Gender': '남자',            
                'Age': 30,                   
                'PaymentMethod': '계좌이체', # Baseline (모든 Payment OHE가 0)
                'Streaming TV': '미가입',     
                'OnlineSecurity': '미가입',   
                'CustomerLTV': 5000,
                'ServiceDuration': 10
            }

            # 3. 실행 (함수 호출)
            generate_recommendations_contrast(
                user_text_input,
                user_features_input,
                lr_model,
                sbert_model,
                corpus_embeddings,
                df_text,
                df_cluster
            )

    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()