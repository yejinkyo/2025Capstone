import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import joblib
import os

# =====================
# 1. 리소스 로드
# =====================
def init_contrastive_resources():
    paths = {
        'lr': 'data/processed/lr_model.joblib',
        'emb': 'data/processed/corpus_embeddings.joblib',
        'text': 'data/processed/telco_narrative_corpus.csv',
        'cluster': 'data/processed/telco_cleaned_data.csv'
    }

    lr_model = joblib.load(paths['lr'])
    corpus_emb = joblib.load(paths['emb'])
    df_text = pd.read_csv(paths['text'])
    df_cluster = pd.read_csv(paths['cluster'])
    sbert = SentenceTransformer('jhgan/ko-sroberta-multitask')
        
    return lr_model, sbert, corpus_emb, df_text, df_cluster

# =====================
# 2. 대조 분석 실행 함수
# =====================
def perform_contrastive_analysis_for_user(user_id, consult_text, resources):
    lr_model, sbert, corpus_emb, df_text, df_cluster = resources
    
    # 실패 시 반환 키를 'role_model_pattern'으로 통일 
    fail_response = {"role_model_pattern": "분석 불가 (데이터 부족)", "insight": "데이터 부족"}

    if sbert is None:
        return fail_response

    try:
        # 1. 유사 고객(Role Model) 찾기
        if not consult_text:
            consult_text = "서비스 불만 및 해지 고민"

        target_emb = sbert.encode([consult_text], normalize_embeddings=True)
        sim_scores = util.cos_sim(target_emb, corpus_emb)[0].numpy()
        best_idx = np.argmax(sim_scores)
        
        b_row = df_text.iloc[best_idx]
        
        # B고객 ID 찾기 
        b_id = None
        for col_name in ['CustomerID', 'customerID', 'id', 'CustomerId']:
            if col_name in b_row:
                b_id = b_row[col_name]
                break
        
        if b_id is None:
            return {"role_model_pattern": "유사 사례 매칭 실패", "insight": "유사 텍스트를 찾았으나 ID 매칭 실패"}

        # 2. 유사 군집 내 롤모델 찾기
        id_col_cluster = None
        for col in ['CustomerID', 'customerID', 'id', 'CustomerId']:
            if col in df_cluster.columns:
                id_col_cluster = col
                break
        
        if not id_col_cluster:
             return {"role_model_pattern": "데이터 ID 컬럼 오류", "insight": "Cluster 데이터 ID 컬럼 확인 필요"}

        # B 고객의 군집 찾기
        # ID 타입 통일 (String)
        b_cluster_row = df_cluster[df_cluster[id_col_cluster].astype(str) == str(b_id)]
        
        if b_cluster_row.empty:
            return {"role_model_pattern": "유사 그룹 탐색 실패", "insight": "유사 고객의 군집 정보 없음"}
            
        cluster_id = b_cluster_row['kmeans_cluster_id'].values[0]
        
        # 이탈하지 않은 이웃들
        # Churn Label 컬럼명 확인
        churn_col = 'Churn Label' if 'Churn Label' in df_cluster.columns else 'ChurnLabel'
        if churn_col not in df_cluster.columns:
             # 라벨이 없으면 그냥 전체 이웃 반환
             neighbors = df_cluster[df_cluster['kmeans_cluster_id'] == cluster_id]
        else:
            neighbors = df_cluster[
                (df_cluster['kmeans_cluster_id'] == cluster_id) & 
                (df_cluster[churn_col] == 'No')
            ]

        # 3. 차이점 분석
        recommendations = []
        target_services = ['OnlineSecurity', 'OnlineBackup', 'TechSupport', 'UnlimitedData', 'StreamingTV']
        
        # 내 데이터 조회
        my_row = df_cluster[df_cluster[id_col_cluster].astype(str) == str(user_id)]
        
        if my_row.empty:
            # 내 데이터가 없으면 이웃들이 많이 쓰는거 그냥 추천
            for col in target_services:
                if col in neighbors.columns:
                    group_usage = neighbors[col].apply(lambda x: 1 if str(x).lower() in ['yes', '1', 'true'] else 0).mean()
                    if group_usage >= 0.4:
                        recommendations.append(f"{col}")
        else:
            # 비교 추천
            for col in target_services:
                if col in my_row.columns and col in neighbors.columns:
                    my_val = my_row[col].values[0]
                    # 나는 안 쓰는데
                    if str(my_val).lower() in ['no', '0', 'false', 'nan']:
                        # 그룹은 많이 쓸 때
                        group_usage = neighbors[col].apply(lambda x: 1 if str(x).lower() in ['yes', '1', 'true'] else 0).mean()
                        if group_usage >= 0.4:
                            recommendations.append(f"{col}")

        if not recommendations:
            return {
                "role_model_pattern": "기본 요금제 유지",
                "insight": "유사 고객들은 현재 상태에 만족하고 있습니다."
            }
        
        return {
            "role_model_pattern": ", ".join(recommendations),
            "insight": f"유사한 만족 고객들은 {recommendations[0]} 등을 이용중입니다."
        }

    except Exception as e:
        print(f"[대조분석] 로직 실행 중 에러: {e}")
        return {"role_model_pattern": "분석 중 기술적 오류", "insight": str(e)}
    

if __name__ == "__main__":
    print("--- 대조 분석 모듈 테스트 시작 ---")
    
    # 1. 리소스 초기화
    resources = init_contrastive_resources()
    
        
    # 2. 테스트 데이터 설정
    test_user_id = "10008" 
    test_text = "데이터 요금이 너무 많이 나와서 부담됩니다. "
    
    print(f"\n2. 분석 실행 (User ID: {test_user_id})")
    print(f"   상담 내용: {test_text}")
    
    # 3. 함수 호출
    result = perform_contrastive_analysis_for_user(test_user_id, test_text, resources)
    
    # 4. 결과 출력
    print("\n3. 분석 결과:")
    print(f"   - Role Model Pattern: {result['role_model_pattern']}")
    print(f"   - Insight: {result['insight']}") 