import pandas as pd
import os

# 1. 데이터 로드 및 병합 (기존 코드 유지)
try:
    df_service = pd.read_csv('data/Rag/Service.csv')
    df_plan = pd.read_csv('data/Rag/Plan.csv')

    df_plan_processed = df_plan[['요금제명', '가격', '상세정보(기본)', '상세정보(토글)']].copy()
    
    # 결측치 처리 및 상세설명 합치기
    df_plan_processed['상세정보(기본)'] = df_plan_processed['상세정보(기본)'].fillna('')
    df_plan_processed['상세정보(토글)'] = df_plan_processed['상세정보(토글)'].fillna('')
    df_plan_processed['상세설명'] = df_plan_processed['상세정보(기본)'] + " " + df_plan_processed['상세정보(토글)']

    df_plan_processed = df_plan_processed.rename(columns={'요금제명': '서비스명', '가격': '요금'})
    df_plan_final = df_plan_processed[['서비스명', '요금', '상세설명']]

    df_merged = pd.concat([df_service, df_plan_final], ignore_index=True)

    OUTPUT_DIR = 'data/processed'
    os.makedirs(OUTPUT_DIR, exist_ok=True) # 폴더가 없으면 생성하도록 수정
    
    output_filename = os.path.join(OUTPUT_DIR, 'Merged_Service_Plan.csv')
    df_merged.to_csv(output_filename, index=False, encoding='utf-8-sig')

    print(f"Merged DataFrame shape: {df_merged.shape}")
    
    # 2. 키워드 매핑 및 카테고리 분류
    # 요청하신 모든 키워드를 포함하도록 맵 확장
    keywords_map = {
        # --- 요청하신 띄어쓰기 없는 버전 ---
        "OnlineSecurity": [
            '스팸', '피싱', '해킹', '안심', '보호', '자녀', '원키퍼', 'Security', '백신', '바이러스'
        ],
        "OnlineBackup": [
            '백업', '보관', '클라우드', '저장', '복원', 'Backup', '마이데이터'
        ],
        "TechSupport": [
            '원격', '매니저', '도우미', '비서', 'Support', '상담', '점검', 'AS', 'A/S', '지원'
        ],
        "UnlimitedData": [
            '무제한', '속도', '데이터', 'Unlimited', 'QoS', '안심옵션'
        ],
        "PaperlessBilling": [
            '청구서', '이메일', '모바일', '앱', 'Paperless', '전자', '명세서'
        ],
        
        # --- 요청하신 한글 결제 관련 ---
        "계좌이체": [
            '계좌', '은행', '자동이체', '출금', '통장'
        ],
        "신용카드": [
            '신용카드', '체크카드', '제휴카드', '카드사', '할부'
        ],

        # --- 요청하신 띄어쓰기 있는 영문 버전 및 나머지 ---
        "Online Security": [
            '스팸', '피싱', '해킹', '안심', '보호', 'Security', '차단', '인증', '도용', '유해'
        ],
        "Online Backup": [
            '백업', '보관', '클라우드', 'Backup', '용량'
        ],
        "Device Protection": [
            '폰교체', '파손', '보험', '분실', 'Protection', '수리', '보상', '케어', '배터리'
        ],
        "Tech Support": [
            '원격', '문의', '전문가', 'Support', '콜', '안내', '해결'
        ],
        "Streaming TV": [
            '티빙', '넷플릭스', '디즈니', '유튜브', '모바일tv', '아이들나라', 'TV', 'OTT', '방송', '실시간'
        ],
        "Streaming Movies": [
            '유플레이', '영화', '비디오', '시네마', 'Movie', 'VOD', '드라마', '콘텐츠', '영상'
        ],
        "Phone Service": [
            '음성', '통화', '전화', 'Phone', 'VoLTE', '발신', '수신'
        ],
        "Multiple Lines": [
            '회선', '결합', '가족', '투폰', '번호', 'Multiple', '쉐어링', '나눠쓰기'
        ],
        "Paperless Billing": [
            '청구서', '탄소', '빌링', '우편', '납부', '영수증'
        ],
        "Internet Service": [
            '인터넷', '와이파이', 'WiFi', 'Giga', '광랜', 'LAN', '속도', 'Internet'
        ],
        "Contract": [
            '약정', '할인반환금', '계약', '해지', '위약금', '개월', '기간', '가입', '유지', '선택약정'
        ],
        "Payment Method": [
            '납부', '결제', '수납', 'Payment', '요금납부', '지로'
        ]
    }

    def assign_categories(row):
        # 서비스명과 상세설명을 모두 검색 대상으로 함
        text_to_search = str(row['서비스명']) + " " + str(row['상세설명'])
        matched_categories = []

        for category, keywords in keywords_map.items():
            for keyword in keywords:
                if keyword in text_to_search:
                    matched_categories.append(category)
                    break # 해당 카테고리가 확인되면 다음 카테고리로 넘어감
        
        # 매칭된 카테고리가 없으면 '기타', 있으면 상위 2개까지 콤마로 연결
        if not matched_categories:
            return "기타"
        return ", ".join(matched_categories[:2])

    df_merged['카테고리'] = df_merged.apply(assign_categories, axis=1)

    SAVE_DIR = 'data/processed'
    os.makedirs(SAVE_DIR, exist_ok=True)
    OUTPUT_FILENAME = 'Final_Mapped_Service_Plan.csv'
    SAVE_PATH = os.path.join(SAVE_DIR, OUTPUT_FILENAME)

    df_merged.to_csv(SAVE_PATH, index=False, encoding='utf-8-sig')

    print(f"✅ Categorization complete. Saved to {SAVE_PATH}")
    print(df_merged[['서비스명', '카테고리']].head(10))

except Exception as e:
    print(f"Error: {e}")