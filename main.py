import gradio as gr
import json
import os
from dotenv import load_dotenv
import time
import pandas as pd
import re
import openai

# ====================================================
# 1. 로컬 모듈 임포트 및 전역 리소스 로딩
# ====================================================

from src.data_generator import generate_analysis_result
from src.llm.msg_generator import generate_marketing_message
from src.model.rec_v2 import init_shap_model
from src.model.rec_v1 import ContrastiveAnalyzer

# css 스타일
with open("src/style/main.css", "r", encoding="utf-8") as css:
    css = css.read()

# .env 파일 로드 및 API 키 설정
load_dotenv()
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ----------------------------------------------------
# 리소스 초기화
# ----------------------------------------------------
shap_model, shap_df_origin = init_shap_model('data/raw/telco2.csv')
GLOBAL_SHAP_RESOURCES = (shap_model, shap_df_origin)
GLOBAL_ANALYZER_INSTANCE = ContrastiveAnalyzer()

RAG_DATA_PATH = 'data/processed/Final_Mapped_Service_Plan.csv'
DF_RAG = pd.read_csv(RAG_DATA_PATH)

# ==========================================
# 2. 분석 및 생성 함수
# ==========================================

def get_rag_info_by_category(recommended_categories):
    """
    분석된 카테고리(영어)에 해당하는 상품들을 검색한 후,
    전체 후보 중 '상위 3개' 서비스만 최종 선별하여 반환합니다.
    """
    
    # 1. 예외 처리
    if DF_RAG.empty:
        return "", ""
    if not recommended_categories:
        return "", ""

    print(f"🔎 [RAG 상품선별] 분석된 카테고리: {recommended_categories}")

    # 2. 전체 후보 상품 모으기 (일단 다 찾음)
    all_matched_rows = pd.DataFrame()

    for category in recommended_categories:
        # 매칭 로직 (소문자/공백 무시)
        target_cat = category.lower().replace(" ", "")
        
        mask = DF_RAG['카테고리'].astype(str).str.lower().str.replace(" ", "").str.contains(target_cat, na=False)
        matches = DF_RAG[mask]
        
        if not matches.empty:
            # 검색된 상품들을 후보군에 추가
            all_matched_rows = pd.concat([all_matched_rows, matches])
    
    # 3. 상품 선별 로직 (핵심)
    rag_html = ""
    rag_text = ""
    
    if not all_matched_rows.empty:
        # (1) 중복 제거 (여러 카테고리에 걸친 상품이 있을 수 있음)
        all_matched_rows = all_matched_rows.drop_duplicates(subset=['서비스명'])
        
        # (2) 딱 3개만 자르기 (CSV 상단에 있는 상품이 우선순위가 높다고 가정)
        # 만약 '요금'이 비싼 순으로 하고 싶다면 여기서 .sort_values() 추가 가능
        final_3_services = all_matched_rows.head(3)
        
        print(f"✅ [RAG 결과] 전체 {len(all_matched_rows)}개 후보 중 상위 3개 선별 완료")

        # (3) 최종 3개 상품에 대해서만 텍스트/HTML 생성
        rag_html += "<div style='margin-bottom: 8px;'><strong>🎁 AI 추천 베스트 상품 (TOP 3)</strong></div>"
        
        for _, row in final_3_services.iterrows():
            # 데이터 추출
            cat_name = row.get('카테고리', '기타')
            service_name = row.get('서비스명', '이름 없음')
            price = row.get('요금', '가격 정보 없음')
            desc = str(row.get('상세설명', ''))[:40] # 설명은 짧게

            # 1) UI용 HTML 생성
            rag_html += f"""
            <div style="font-size: 12px; color: #4b5563; margin-bottom: 6px; padding: 8px; border: 1px solid #e5e7eb; border-radius: 6px; background: #fff;">
                <div style="font-size: 10px; color: #6366f1; font-weight: bold; margin-bottom: 2px;">{cat_name}</div>
                <div style="font-weight:700; color:#1f2937; font-size: 13px;">{service_name}</div>
                <div style="color:#059669; font-size: 12px; font-weight: 600;">{price}</div>
                <div style="color:#9ca3af; font-size: 11px; margin-top: 2px;">{desc}...</div>
            </div>
            """
            
            # 2) LLM 전달용 텍스트 생성 (상품명 리스트)
            rag_text += f"👉 {service_name} ({price})\n"
            
    else:
        print("⚠️ [RAG 실패] 매칭되는 상품이 하나도 없습니다.")
        rag_text = "(추천 상품 없음)"

    return rag_html, rag_text

import re

def parse_generated_text(text):
    """
    LLM 출력을 버전별로 분리하여 순수한 텍스트 리스트로 반환하는 함수 (강화된 Regex)
    """
    # 분리할 패턴 정의 (버전 헤더를 찾아서 자름)
    pattern = r"(?:^|\n)(?:\*\*|\[|#+\s)?(?:버전|옵션|Version|Option)\s?\d+.*?(?:\*\*|\]|:)?"
    
    # 텍스트 분리
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    
    # 공백 제거 및 필터링
    options = []
    for p in parts:
        clean_p = p.strip()
        # 내용이 너무 짧은 경우(헤더만 남은 경우 등) 제외하고 추가
        if len(clean_p) > 10: 
            options.append(clean_p)
    
    # 만약 분리에 실패했다면(옵션이 1개도 안 나오면)
    if not options:
        # 비상 대책: "버전" 키워드가 포함된 줄을 기준으로 강제 분리 시도
        if "버전" in text:
             return [t.strip() for t in text.split("버전") if len(t.strip()) > 10]
        return [text] # 그래도 안 되면 통째로 반환

    return options

def generate_short_suggestion(consult_text, rec_services, rag_info, api_key):
    """대시보드용 짧은 AI 제안 생성 (LLM 호출)"""
    if not api_key: return "API 키를 입력해주세요."

    prompt = f"""
    당신은 통신사 AI 어시스턴트입니다.
    고객의 상담 내용과 추천 서비스 정보를 바탕으로, **가장 시급한 해결책을 한 문장으로 요약**해서 제안하세요.

    [고객 상담] "{consult_text}"
    [추천 서비스] {', '.join(rec_services)}
    [실제 상품 정보] {rag_info}

    [작성 규칙]
    1. **한 문장(50자 이내)**으로 간결하게 작성하세요.
    2. 고객의 불만(상담내용)을 해결하는 구체적인 상품명(실제 상품 정보 참고)을 언급하세요.
    3. 이모지를 사용하여 눈에 띄게 만드세요.
    """

    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI 제안 생성 실패: {str(e)}"

# ==========================================
# 3. 분석 및 생성 함수
# ==========================================

def analyze_customer(user_id, consult_text, new_user_features_json, api_key_input):
    """
    [분석 버튼] 클릭 시 실행. 신규 유저 JSON 입력 시 해당 피처로 분석을 시도함.
    """
    if not user_id:
        return None, "<div>ID를 입력해주세요</div>", "<div>ID를 입력해주세요</div>", "<div>ID를 입력해주세요</div>","<div>ID 입력 필요</div>",gr.update(interactive=False)

    # 1. 신규 유저 피처 JSON 파싱
    user_features_input = None
    if new_user_features_json and "{" in new_user_features_json:
        try:
            user_features_input = json.loads(new_user_features_json)
            print(f"신규 유저 로드: ID {user_id}")
        except json.JSONDecodeError as e:
            error_html = f"<div class='dashboard-card'><div class='empty-state text-red-700'>JSON 파싱 오류: {e}</div></div>"
            return None, error_html, error_html, error_html, gr.update(interactive=False)

    try:
        # 1. 분석 로직 실행
        json_str = generate_analysis_result(
            user_id=user_id,
            consult_text=consult_text,
            shap_resources=GLOBAL_SHAP_RESOURCES,
            analyzer_instance=GLOBAL_ANALYZER_INSTANCE,
            user_features_input=user_features_input 
        )
        data = json.loads(json_str)
        
        if "error" in data:
            error_html = f"<div class='dashboard-card'><div class='empty-state text-red-700'>{data['error']}</div></div>"
            return None, error_html, error_html, error_html, error_html, gr.update(interactive=False)

        # 3. 데이터 추출: profile, shap_res, contrast_res
        profile = data.get('customer_profile', {})
        shap_res = data.get('analysis_results', {}).get('shap_analysis', {})
        contrast_res = data.get('analysis_results', {}).get('contrastive_analysis', {})
        
        # 4. RAG 정보 생성
        list_contrast = contrast_res.get('recommended_services', [])
        list_shap = shap_res.get('recommended_actions', [])

        rec_cats = list(set(list_contrast + list_shap))

        # 실제 상품 정보 검색
        rag_html_content, rag_text_content = get_rag_info_by_category(rec_cats)
        
        # 나중에 문구 생성할 때 쓰기 위해 data 딕셔너리에 저장
        data['rag_context'] = rag_text_content

        # LLM 짧은 제안 생성
        ai_suggestion = generate_short_suggestion(consult_text, rec_cats, rag_text_content, api_key_input)

        # 이탈 확률 데이터 타입 처리
        churn_prob_val = contrast_res.get('churn_probability', 0.0)
        if isinstance(churn_prob_val, str):
             try:
                 churn_val = float(churn_prob_val.replace('%', '').strip())
             except ValueError:
                 churn_val = 0.0
        else:
            churn_val = float(churn_prob_val)
        churn_val = churn_val * 100

        # 4. [UI: score_html] 이탈 확률
        risk_badge = "위험" if churn_val >= 70 else "주의" if churn_val >= 50 else "안정"
        badge_class = "badge-critical" if churn_val >= 70 else "badge-warning" if churn_val >= 50 else "badge-safe"
        bar_color = "#ef4444" if churn_val >= 70 else "#f59e0b" if churn_val >= 50 else "#10b981"
        
        score_html = f"""
        <div class="card-content">
            <div class="card-title">이탈 확률</div>
            <div class="churn-score-container">
                <span class="churn-score-text" style="color: {bar_color}">{churn_val:.1f}%</span>
                <span class="churn-badge {badge_class}">{risk_badge}</span>
            </div>
            <div class="progress-bg">
                <div class="progress-bar" style="width: {churn_val}%; background-color: {bar_color};"></div>
            </div>
        </div>
        """

        # SHAP 추천 (방어 요인)
        risk_factors_html = ""
        shap_actions = shap_res.get('recommended_actions', [])
        if not shap_actions:
            risk_factors_html = "<div class='text-gray-400 text-sm p-2'>위험 요인이 없습니다.</div>"
        else:
            for idx, item in enumerate(shap_actions):
                risk_factors_html += f"""
                <div class="factor-item bg-red-50 text-red-700" style="border-left: 3px solid #ef4444;">
                    <div class="factor-icon">🚨</div>
                    <div class="factor-text">
                        <span class="factor-label">위험 요인 {idx+1}</span>
                        <span class="factor-desc">{item}</span>
                    </div>
                </div>
                """
            
        # 대조분석 추천
        opportunity_html = ""
        rec_services = contrast_res.get('recommended_services', [])
        for idx, item in enumerate(rec_services):
            opportunity_html += f"""
            <div class="factor-item bg-blue-50 text-blue-700" style="border-left: 3px solid #3b82f6;">
                <div class="factor-icon">💎</div>
                <div class="factor-text">
                    <span class="factor-label">추천 서비스 {idx+1}</span>
                    <span class="factor-desc">{item}</span>
                </div>
            </div>
            """

        # AI 제안 HTML 구성 (LLM 멘트 + RAG 상품 정보)
        ai_proposal_html = f"""
        <div class="card-content" style="height: 100%; background-color: #f0fdf4; border: 1px solid #bbf7d0;">
            <div style="font-size: 14px; color: #15803d; margin-bottom: 12px; line-height: 1.5;">
                🤖 <strong>AI의 한마디:</strong><br>{ai_suggestion}
            </div>
            <div style="background: white; padding: 8px; border-radius: 6px; border: 1px solid #e5e7eb;">
                <div style="font-size: 12px; font-weight: bold; color: #4b5563; margin-bottom: 6px; border-bottom: 1px solid #eee; padding-bottom: 4px;">🎁 추천 상품 상세</div>
                {rag_html_content if rag_html_content else "<div style='color:#9ca3af; font-size:11px;'>매칭된 상품 없음</div>"}
            </div>
        </div>
        """
        
        analysis_html = f"""
        <div class="card-content">
            
            <!-- 섹션 1: 이탈 위험 요인 -->
            <div style="margin-bottom: 16px;">
                <h4 style="font-size: 13px; color: #ef4444; margin-bottom: 8px; font-weight: 700;">📉 주요 이탈 요인</h4>
                <div class="factors-list" style="max-height: 200px; overflow-y: auto;">
                    {risk_factors_html}
                </div>
            </div>

            <!-- 섹션 2: 추천 서비스 -->
            <div style="margin-bottom: 16px;">
                <h4 style="font-size: 13px; color: #3b82f6; margin-bottom: 8px; font-weight: 700;">📈 다른 고객이 이용하는 서비스</h4>
                <div class="factors-list" style="max-height: 140px; overflow-y: auto;">
                    {opportunity_html}
                </div>
            </div>

            <!-- 섹션 3: 실제 상품 제안 -->
            <div style="margin-top: 16px; padding: 10px; background: #fff; border: 1px solid #e5e7eb; border-radius: 8px;">
                <h4 style="color: #4b5563; font-size: 13px; margin-bottom: 8px;">🎁 LGU+ 실제 상품 제안</h4>
                <div class="factors-list" style="max-height: 140px; overflow-y: auto;">
                    {ai_proposal_html}
                </div>
            </div>

            <!-- 하단 인사이트 박스 -->
            <div class="recommendation-box">
                💡 <strong>마케팅 전략:</strong> {contrast_res.get('insight_message')}
            </div>
        </div>
        """
        
        # 6. [UI: profile_html] 프로필 업데이트 
        profile_html = f"""
        <div class="dashboard-card">
            <div class="profile-header">
                <div class="avatar">👩🏻</div>
                <div class="user-name">{profile.get('customer_id')}</div>
                <div class="user-id">성별: {profile.get('gender')}</div>
            </div>
            <div class="stat-row">
                <span class="stat-label">가입 기간</span>
                <span class="stat-value">{profile.get('tenure_month')}개월</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">월 요금</span>
                <span class="stat-value">${profile.get('monthly_charge'):.2f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">TV 스트리밍</span>
                <span class="stat-value">{profile.get('streaming_tv')}</span>
            </div>
        </div>
        """

        return data, score_html, analysis_html, profile_html, gr.update(interactive=True)

    except Exception as e:
        err_msg = f"분석 오류: {str(e)}"
        print(f"Analyze Customer Exception: {e}")
        return None, f"<div>{err_msg}</div>", f"<div>{err_msg}</div>", f"<div>{err_msg}</div>", gr.update(interactive=False)
    
import re

def parse_generated_text(text):
    """LLM 출력을 개별 메시지 옵션으로 분리하는 헬퍼 함수"""
    
    # 정규식 패턴 수정: 줄바꿈 조건 (?:^|\n)을 제거하고, 구분자 패턴을 명확히 합니다.
    # 텍스트 전체에서 '**[버전 N:' 형태를 찾습니다.
    # **와 ]**를 기준으로 패턴을 명확히 정의합니다.
    # ( ) 괄호로 패턴을 캡처 그룹으로 만들어, re.split 시 이 패턴도 결과에 포함되게 합니다.
    pattern = r"(\*\*\[버전\s?\d+.*?\]\*\*)" 

    # 캡처 그룹을 사용하여 분할하면, parts 리스트는 [잔여물, 패턴1, 내용1, 패턴2, 내용2, ...] 순이 됩니다.
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    options = []
    
    # LLM이 출력한 전체 텍스트에서 버전 제목과 내용을 분리하여 options 리스트에 추가합니다.
    for i in range(1, len(parts)):
        # 홀수 인덱스 = 버전 제목 (패턴)
        if i % 2 == 1:
            header = parts[i].strip()
        # 짝수 인덱스 = 버전 내용 (본문)
        elif i % 2 == 0:
            content = parts[i].strip()
            
            # 내용이 너무 짧은 경우 제외 (짧은 잔여물 필터링)
            if header and content and len(content) > 10: 
                # 헤더와 내용을 합쳐 하나의 옵션으로 만듭니다. (사용자가 어떤 버전인지 알 수 있도록)
                # 옵션 텍스트에서 불필요한 마크다운 기호들을 제거하는 로직이 필요할 수 있습니다.
                final_option = f"{header.strip('*[] ')} - {content.strip()}"
                options.append(final_option) 

    # 파싱이 성공하지 못하면 원본 텍스트를 반환합니다.
    if len(options) < 2:
        return [text] 

    return options

def generate_message_action(user_data_state, consult_text, api_key):
    """
    [문구 생성] 버튼 클릭 시 LLM 호출
    """
    if not user_data_state:
        return gr.update(visible=False), "⚠️ 먼저 [분석]을 실행해주세요."
    
    # 1. State에서 RAG 정보 꺼내기
    rag_info_text = user_data_state.get('rag_context', "")
    
    # 디버깅: 실제로 넘어왔는지 확인
    if not rag_info_text:
        print("⚠️ [경고] 문구 생성 시 RAG 정보가 비어있습니다!")
    else:
        print(f"🚀 [LLM 전송] RAG 정보 포함됨 (길이: {len(rag_info_text)})")

    # 2. API Key 처리
    real_api_key = api_key if api_key else os.getenv("OPENAI_API_KEY", "")
    
    # 3. LLM 호출 (src/llm/msg_generator.py)
    full_message = generate_marketing_message(
        analysis_json=user_data_state, 
        consult_text=consult_text, 
        api_key=real_api_key, 
        rag_info_text=rag_info_text  # <--- 여기서 확실히 넘겨줌
    )
    
    if "오류" in full_message or "API Key" in full_message:
        return gr.update(visible=False), full_message

    candidates = parse_generated_text(full_message)
    return gr.update(choices=candidates, value=None, visible=True), "✅ 마케팅 문구가 생성되었습니다."

def display_selected_message(selected_text):
    """
    [라디오 버튼 선택] 시 선택된 문구를 HTML 말풍선으로 표시
    """
    if not selected_text:
        return "", "" # 선택 해제 시 빈 값

    chat_bubble_html = f"""
    <div style="background-color: #f3f4f6; padding: 20px; border-radius: 12px; margin-top: 10px; border: 1px solid #e5e7eb;">
        <div class="chat-bubble-ai" style="display: flex; gap: 12px; align-items: flex-start;">
            <div class="chat-avatar" style="width: 36px; height: 36px; background: #4f46e5; color: white; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-weight: bold;">AI</div>
            <div class="chat-text" style="background: white; padding: 16px; border-radius: 0 16px 16px 16px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); line-height: 1.6; color: #374151; width: 100%;">
                {selected_text.replace(chr(10), '<br>')}
            </div>
        </div>
        <div style="text-align: right; margin-top: 8px;">
            <span style="font-size: 11px; color: #9ca3af;">전송 미리보기 • 방금 전</span>
        </div>
    </div>
    """
    return chat_bubble_html, selected_text # HTML과 텍스트(전송용) 반환

def send_message_action(message_text):
    if not message_text:
        return "전송할 메시지가 없습니다."
        
    time.sleep(0.5)
    return f"메시지가 성공적으로 전송되었습니다!"



# ==========================================
# 4. Gradio UI 구성
# ==========================================
# 테스트 유저 데이터
NEW_USER_DEFAULT_FEATURES = {
    "CustomerId": "0001",
    "Gender": "남자",
    "Age": 30,
    "Married": "Yes",
    "Dependents": "No",
    "Tenure_month": 1,
    "Monthly_charge": 110.00,
    "Sum_charge": 110.00,
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "TechSupport": "No",
    "UnlimitedData": "Yes",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "신용카드",
    "Device Protection": "No",
    "Contract": "Yes",
    "StreamingTV": "No",
    "ConsultText": "요금이 너무 비싸고 느려서 해지하고 싶어요."
}
NEW_USER_DEFAULT_JSON_STRING = json.dumps(NEW_USER_DEFAULT_FEATURES, indent=4)


with gr.Blocks() as demo:
    
    # 상태 저장소
    user_state = gr.State({})
    message_state = gr.State("")
    api_key = gr.State(DEFAULT_API_KEY)

    # 헤더
    with gr.Row():
        gr.Markdown("# 🛡️ 고객 이탈 방어 & 초개인화 마케팅 대시보드")

        # 유저 피처 입력 영역
    with gr.Accordion("유저 데이터", open=False):
        input_feature_json = gr.Textbox(
            label="신규 유저 피처 데이터 (JSON)",
            value=NEW_USER_DEFAULT_JSON_STRING,
            lines=20,
            placeholder="여기에 고객의 모든 데이터를 JSON 형태로 입력하세요."
        )

    # --- 컨트롤 ---
    with gr.Row(variant="panel"):
        with gr.Column(scale=2):
            input_user_id = gr.Textbox(label="고객 ID", value="0001", placeholder="고객 ID를 입력하세요.")
        with gr.Column(scale=2):
            input_consult_text = gr.Textbox(label="최근 상담", value="요금이 너무 비싸고 느려서 해지하고 싶어요.", placeholder="상담 내용 입력")
        with gr.Column(scale=1):
            btn_analyze = gr.Button("🔍 분석", variant="primary", size="lg")
        
    # --- [상단] 대시보드 영역 ---
    with gr.Row(equal_height=True):
        
        # 1. 고객 정보 카드
        with gr.Column(scale=1):
            user_profile_html = gr.HTML(
                """
                <div class="dashboard-card">
                    <div class="profile-header">
                        <div class="avatar">👤</div>
                        <div class="user-name">고객명</div>
                        <div class="user-id">ID를 입력하세요.</div>
                    </div>
                </div>
                """
            )
            
        # 2. 이탈 스코어 카드
        with gr.Column(scale=1):
            churn_score_html = gr.HTML(
                """
                <div class="dashboard-card">
                    <div class="card-title">CHURN RISK SCORE</div>
                    <div class="empty-state">'분석 실행' 버튼을 눌러<br>이탈 확률을 확인하세요.</div>
                </div>
                """
            )

        # 3. AI 맞춤 제안 & 상품
        with gr.Column(scale=1):
            ai_analysis_html = gr.HTML(
                """
                <div class="dashboard-card">
                    <div class="empty-state">'분석 실행' 버튼을 눌러<br>AI 맞춤 제안 & 상품을 확인하세요.</div>
                </div>
                """
            )

    # --- [하단] 액션 영역 ---
    gr.Markdown("# 📱 마케팅 문구 추천")
    
    with gr.Row():
        # [왼쪽] 마케팅 문구 생성 버튼
        with gr.Column(scale=1):
            gr.Markdown("#### 1. 메시지 생성")
            btn_gen_msg = gr.Button("✨ 마케팅 문구 생성 (AI)", variant="primary")

            # 생성된 문구 리스트
            gr.Markdown("#### 2. 마케팅 문구 추천")
            msg_options_radio = gr.Radio(choices=[], visible=True, interactive=True)
            

        # [오른쪽] 선택된 문구 전송
        with gr.Column(scale=2):
            gr.Markdown("#### 3. 메시지 전송")
            message_display_html = gr.HTML(label="미리보기")

            with gr.Row():
                send_status = gr.Textbox(label="전송 상태", interactive=False, scale=2)
                btn_send_msg = gr.Button("🚀 메시지 전송", variant="secondary", scale=1, interactive=False)

    # ==========================================
    # 5. 이벤트 연결
    # ==========================================

    # 1) 분석 버튼 클릭
    btn_analyze.click(
        fn=analyze_customer,
        inputs=[input_user_id, input_consult_text, input_feature_json, api_key],
        outputs=[user_state, churn_score_html, ai_analysis_html, user_profile_html, btn_send_msg]
    )

    # 2) 문구 생성 버튼 클릭
    btn_gen_msg.click(
        fn=generate_message_action,
        inputs=[user_state, input_consult_text, api_key],
        outputs=[msg_options_radio, send_status]
    )

    # 3) 라디오 버튼 선택 시
    msg_options_radio.change(
        fn=display_selected_message,
        inputs=[msg_options_radio],
        outputs=[message_display_html, message_state]
    ) .then(
        lambda: gr.update(interactive=True), None, [btn_send_msg]
    )

    # 4) 메시지 전송 버튼 클릭
    btn_send_msg.click(
        fn=send_message_action,
        inputs=[message_state],
        outputs=[send_status]
    ) .then(
        lambda: gr.update(interactive=False), None, [btn_send_msg]
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=css)