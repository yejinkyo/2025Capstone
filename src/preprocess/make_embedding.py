import pandas as pd
import joblib
import os
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. 경로 설정
# ==========================================
# 텍스트 데이터가 있는 CSV 파일 경로
TEXT_DATA_PATH = 'data/processed/telco_narrative_corpus.csv'

# 만들어진 임베딩을 저장할 경로
SAVE_PATH = 'data/processed/corpus_embeddings.joblib'

def create_and_save_embeddings():
    print("🚀 코퍼스 임베딩 생성 작업을 시작합니다...")

    # -----------------------------------------------------
    # 2. 데이터 로드 및 'text' 열 추출 (핵심!)
    # -----------------------------------------------------
    if not os.path.exists(TEXT_DATA_PATH):
        print(f"❌ 오류: 데이터 파일({TEXT_DATA_PATH})을 찾을 수 없습니다.")
        return

    try:
        # CSV 파일 전체 읽기
        df = pd.read_csv(TEXT_DATA_PATH)
        
        # 'text' 컬럼이 있는지 확인
        if 'text' not in df.columns:
            print(f"❌ 오류: CSV 파일에 'text' 컬럼이 없습니다.")
            print(f"   - 현재 컬럼 목록: {list(df.columns)}")
            return

        # ★ [중요] 오직 'text' 열만 선택하여 리스트로 변환 ★
        # 결측치(NaN)가 있으면 빈 문자열("")로 채워서 에러 방지
        sentences = df['text'].fillna("").astype(str).tolist()
        
        print(f"  - 데이터 로드 성공!")
        print(f"  - 임베딩할 문장 개수: {len(sentences)}개")
        print(f"  - 첫 번째 문장 예시: {sentences[0][:50]}...")

    except Exception as e:
        print(f"❌ 데이터 처리 중 오류 발생: {e}")
        return

    # -----------------------------------------------------
    # 3. S-BERT 모델 로드 및 변환
    # -----------------------------------------------------
    print("\n📦 S-BERT 모델 로딩 중...")
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')

    print("⏳ 임베딩 변환 중... (문장 6000개 기준 약 1~2분 소요)")
    # 여기서 위에서 뽑은 sentences(텍스트 리스트)만 들어갑니다.
    embeddings = model.encode(sentences, normalize_embeddings=True, show_progress_bar=True)

    # -----------------------------------------------------
    # 4. 파일로 저장
    # -----------------------------------------------------
    # 저장 폴더가 없으면 생성
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    
    joblib.dump(embeddings, SAVE_PATH)
    print(f"\n✅ 임베딩 저장 완료!")
    print(f"  - 저장 위치: {SAVE_PATH}")
    print(f"  - 이제 main.py를 실행하면 이 파일을 바로 불러옵니다.")

if __name__ == "__main__":
    create_and_save_embeddings()