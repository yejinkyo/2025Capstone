import pandas as pd
import os

df_service = pd.read_csv('data/Rag/Service.csv')
df_plan = pd.read_csv('data/Rag/Plan.csv')

df_plan_processed = df_plan[['요금제명', '가격', '상세정보(기본)', '상세정보(토글)']].copy()

df_plan_processed['상세정보(기본)'] = df_plan_processed['상세정보(기본)'].fillna('')
df_plan_processed['상세정보(토글)'] = df_plan_processed['상세정보(토글)'].fillna('')
df_plan_processed['상세설명'] = df_plan_processed['상세정보(기본)'] + " " + df_plan_processed['상세정보(토글)']

df_plan_processed = df_plan_processed.rename(columns={'요금제명': '서비스명', '가격': '요금'})

df_plan_final = df_plan_processed[['서비스명', '요금', '상세설명']]

df_merged = pd.concat([df_service, df_plan_final], ignore_index=True)

OUTPUT_DIR = 'data/processed'

output_filename = os.path.join(OUTPUT_DIR, 'Merged_Service_Plan.csv')
df_merged.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"Merged DataFrame shape: {df_merged.shape}")
print(df_merged.head())
print(df_merged.tail())

try:
    df_merged = pd.read_csv('data/processed/Merged_Service_Plan.csv')

    keywords_map = {
        "Online Security": [
            '스팸', '피싱', '해킹', '안심', '보호', '자녀', '원키퍼', '명의도용', '정보', 'Security', 
            '백신', '차단', '금융사기', '인증', '도용', '유해'
        ],
        "Online Backup": [
            '백업', '보관', '원키퍼', '마이데이터', 'Backup', '클라우드', '저장', '복원'
        ],
        "Device Protection": [
            '폰교체', '파손', '보험', '분실', 'Protection', '수리', '보상', '케어', '배터리', '교체'
        ],
        "Tech Support": [
            '원격', '매니저', '도우미', '비서', 'Support', '상담', '점검', '문의', '전문가', 
            '콜', '안내', 'AS', 'A/S', '지원', '해결'
        ],
        "Streaming TV": [
            '티빙', '넷플릭스', '디즈니', '유튜브', '모바일tv', '아이들나라', 'TV', 'OTT', 
            '방송', '실시간', '채널'
        ],
        "Streaming Movies": [
            '유플레이', '영화', '비디오', '시네마', 'Movie', 'VOD', '드라마', '콘텐츠', '영상'
        ],
        "Paperless Billing": [
            '청구서', '탄소', '이메일', '빌링', '명세서', '우편', '앱', '모바일', 'Paperless', 
            '전자', '납부', '영수증'
        ],
        "Contract": [
            '약정', '할인반환금', '계약', '해지', '위약금', '개월', '기간', '가입', '유지', '선택약정'
        ]
    }

    def assign_categories(row):
        text_to_search = str(row['서비스명']) + " " + str(row['상세설명'])
        matched_categories = []

        for category, keywords in keywords_map.items():
            for keyword in keywords:
                if keyword in text_to_search:
                    matched_categories.append(category)
                    break

        return ", ".join(matched_categories[:2])
    
    df_merged['카테고리'] = df_merged.apply(assign_categories, axis=1)

    SAVE_DIR = 'data/processed'
    os.makedirs(SAVE_DIR, exist_ok=True)
                
    OUTPUT_FILENAME = 'Final_Mapped_Service_Plan.csv'
    SAVE_PATH = os.path.join(SAVE_DIR, OUTPUT_FILENAME)

    df_merged.to_csv(OUTPUT_FILENAME, index=False, encoding='utf-8-sig')

    print(f"✅ Categorization complete. Saved to {OUTPUT_FILENAME}")
    print(df_merged[['서비스명', '카테고리']].head(10))

except Exception as e:
    print(f"Error: {e}")