import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors

def preprocess_telco(input_path: str, output_path: str = None, visualize: bool = False):
    """
    텔코 데이터 전처리 함수
    - 파생 변수 생성
    - 결측치 처리
    - 선택적 시각화 및 CSV 저장
    """
    # ===== 1. 데이터 불러오기 =====
    df = pd.read_csv(input_path,encoding="cp949")

    # ===== 2. 날짜형 변환 & 서비스 이용 기간 =====
    df['StartDate'] = pd.to_datetime(df['StartDate'])
    df['EndDate'] = pd.to_datetime(df['EndDate'])
    df['ServiceDuration'] = (df['EndDate'] - df['StartDate']).dt.days

    # ===== 3. 파생 변수 생성 =====
    df['CLTV_monthly'] = df['CustomerLTV'] / df['ServiceDuration']

     #민주
    df['TotalOtherCharges'] = df['TotalExtraDataCharge'] + df['TotalRoamCharge']
    df['LTVPerSatis'] = df['CustomerLTV'] / df['SatisScore']

    # ===== 4. 결측치 처리 =====
    # 수치형: 평균 대체
    num_cols = df.select_dtypes(include=[np.number]).columns
    imputer_num = SimpleImputer(strategy='mean')
    df[num_cols] = imputer_num.fit_transform(df[num_cols])

    # 범주형: 최빈값 대체
    cat_cols = df.select_dtypes(include=['object']).columns
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

    if visualize:
        plt.figure(figsize=(12, 6))
        sns.histplot(df['CLTV_monthly'], kde=True)
        plt.title("Monthly CLTV Distribution")
        plt.show()

    # ===== 5. 결과 저장 =====
    if output_path:
        df.to_csv(output_path, index=False)
    else:
        print("전처리 완료, 저장 안 함")

    print("최종 데이터 shape:", df.shape)
    return df

def cluster_customers(input_path: str, n_clusters: int = 3, kmeans_name_map: dict = None,
                      dbscan_eps: float = None, dbscan_min_samples: int = None,
                      output_path: str = None, visualize: bool = True):
    """
    K-Means + DBSCAN 군집화
    """
    df = pd.read_csv(input_path)

    # 수치형 변수 선택 (타깃 제외)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'ChurnLabel' in num_cols:
        num_cols.remove('ChurnLabel')
    X = df[num_cols]

    # 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ------------------
    # 1️. K-Means 군집
    # ------------------
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['kmeans_cluster_id'] = kmeans.fit_predict(X_scaled)

    if kmeans_name_map:
        df['cluster_name'] = df['kmeans_cluster_id'].map(kmeans_name_map)

    # ------------------
    # 2️. DBSCAN 군집
    # ------------------
    # eps와 min_samples 계산 참고
    if dbscan_eps is None or dbscan_min_samples is None:
        k = 2 * X_scaled.shape[1]
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(X_scaled)
        distances, _ = neighbors_fit.kneighbors(X_scaled)
        distances = np.sort(distances[:, k-1])
        plt.figure(figsize=(8,5))
        plt.plot(distances)
        plt.title('K-distance Plot (DBSCAN eps 선택 참고)')
        plt.xlabel('데이터 포인트')
        plt.ylabel(f'{k}번째 이웃 거리')
        plt.grid(True)
        plt.show()

    if dbscan_eps is not None and dbscan_min_samples is not None:
        dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        df['dbscan_cluster_id'] = dbscan.fit_predict(X_scaled)

    # ------------------
    # 3️. 저장 및 시각화
    # ------------------
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"군집화 결과 저장: '{output_path}'")

    if visualize:
        plt.figure(figsize=(10,6))
        sns.scatterplot(x=df['ServiceDuration'], y=df['CLTV_monthly'],
                        hue=df['kmeans_cluster_id'], palette='Set2')
        plt.title("Customer Clusters (K-Means)")
        plt.show()

    return df



if __name__ == "__main__":
    preprocess_telco(
        input_path="data/raw/telco.csv",
        output_path="data/processed/telco_cleaned_data.csv",
        visualize=True
    )

    # K-Means 이름 매핑
    cluster_name_map = {
        0: '표준 단기 고객 (월정액)',
        1: '알뜰형 장기 고객 (2년 약정, 저CLTV)',
        2: '기술선호형 중장기 고객 (월정액)',
        3: '초단기 신규 고객 (최대 이탈 위험군)',
        4: '표준 중기 고객 (1년 약정)',
        5: '고가치 장기 고객 (월정액, 고CLTV)',
        6: '알뜰형 중기 고객 (1년 약정)',
        7: '충성도 높은 중기 고객 (월정액)',
        8: '우량 중장기 고객 (1년 약정)',
        9: '초우량 VIP (2년 약정, 최고CLTV)',
        10: '가성비 중시 장기 고객 (2년 약정)',
        11: '기술선호형 장기 고객 (1년 약정)',
        12: '초알뜰 중장기 고객 (2년 약정)',
        13: '신규 장기 약정 고객 (저CLTV)',
        14: '기술선호형 중기 고객 (월정액)',
        15: '장기 월정액 고객 (광랜 선호)',
        16: '표준 단기 고객 (저CLTV)',
        17: '알뜰형 단기 고객 (1년 약정)',
        18: '표준 월정액 전환 고객',
        19: '저가 서비스 장기 고객 (2년 약정)',
        20: '안정적인 우량 고객 (1년 약정)',
        21: '기술민감형 장기 고객 (월정액, 고CLTV)',
        22: '안정적인 중기 고객 (1년 약정)',
        23: '저가 서비스 장기 고객 (1년 약정)',
        24: '다서비스 장기 고객 (2년 약정)',
        25: '단기 신규 고객 (전자결제 선호)',
        26: '우량 장기 고객 (2년 약정)',
        27: '초알뜰 초장기 고객 (2년 약정, 전화 중심)',
        28: '초절약 단일 서비스 신규 고객 (전화만 사용)',
        29: '안정적인 장기 고객 (월정액)'
    }

    # 군집화
    clustered_df = cluster_customers(
        input_path="data/processed/telco_cleaned_data.csv",
        n_clusters=6,
        kmeans_name_map=cluster_name_map,
        dbscan_eps=None,          # eps 결정 전 None
        dbscan_min_samples=None,  # min_samples 결정 전 None
        output_path="data/processed/telco_cleaned_data.csv",
        visualize=True
    )

    #하이
    #감사합니당