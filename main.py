import gradio as gr
import json
import os
from dotenv import load_dotenv
import time

# 로컬 모듈 임포트
from src.data_generator import generate_analysis_result
from src.model.rec_v2 import init_shap_model
from src.model.rec_v1 import init_contrastive_resources
from src.llm.msg_generator import generate_marketing_message

# .env 파일 로드 (API 키가 있다면)
load_dotenv()
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ==========================================
# 0. 리소스 초기화
# ==========================================
shap_resources = init_shap_model('data/raw/telco2.csv') 
contrast_resources = init_contrastive_resources()

# ==========================================
# 1. 분석 및 생성 함수
# ==========================================

def analyze_customer(user_id, consult_text):
    """
    [분석 버튼]
    """
    if not user_id:
        return None, "<div>ID를 입력해주세요</div>", "<div>ID를 입력해주세요</div>", gr.update(interactive=False)

    try:
        # 1. 실제 분석 로직 호출
        json_str = generate_analysis_result(user_id, consult_text, shap_resources, contrast_resources)
        data = json.loads(json_str)
        
        if "error" in data:
            error_html = f"<div class='dashboard-card'><div class='empty-state text-red-700'>{data['error']}</div></div>"
            return None, error_html, error_html, gr.update(interactive=False)

        # 2. 데이터 파싱
        profile = data.get('customer_profile', {})
        shap_res = data.get('analysis_results', {}).get('shap_analysis', {})
        contrast_res = data.get('analysis_results', {}).get('contrastive_analysis', {})
        
        # 숫자 데이터 처리
        churn_str = shap_res.get('churn_probability', '0%')
        churn_val = float(churn_str.replace('%', ''))
        
        # 3. [UI] 이탈 스코어 HTML 생성
        risk_badge = "위험 (Critical)" if churn_val >= 70 else "주의 (Warning)" if churn_val >= 50 else "안정 (Safe)"
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

        # 4. [UI] AI 분석 원인 HTML 생성 (추천 서비스 리스트업)
        factors_html = ""
        
        # SHAP 추천 (방어 요인) -> 빨간색 배경
        for idx, item in enumerate(shap_res.get('recommended_actions', [])):
            factors_html += f"""
            <div class="factor-item bg-red-50 text-red-700">
                <div class="factor-icon">🚨</div>
                <div class="factor-text">
                    <span class="factor-label">이탈 방어 {idx+1}</span>
                    <span class="factor-desc">{item}</span>
                </div>
            </div>
            """
            
        # 대조분석 추천 (추가 제안) -> 노란색 배경
        for idx, item in enumerate(contrast_res.get('recommended_services', [])):
            factors_html += f"""
            <div class="factor-item bg-yellow-50 text-yellow-700">
                <div class="factor-icon">💡</div>
                <div class="factor-text">
                    <span class="factor-label">추가 제안 {idx+1}</span>
                    <span class="factor-desc">{item}</span>
                </div>
            </div>
            """
        
        # 최종 추천 문구 추출
        final_recs = data.get('marketing_objective', {}).get('final_recommendations', [])
        rec_text = ", ".join(final_recs[:2]) if final_recs else "맞춤 혜택"

        analysis_html = f"""
        <div class="card-content">
            <div class="card-title">🟣 AI 분석 인사이트</div>
            <div class="factors-list">
                {factors_html if factors_html else "<div class='empty-state'>특이사항 없음</div>"}
            </div>
            <div class="recommendation-box">
                 💡 <strong>AI 전략:</strong> {contrast_res.get('insight_message', '데이터 기반 맞춤 제안')}
            </div>
        </div>
        """
        
        # 5. [UI] 프로필 업데이트 (실제 데이터 반영)
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
        </div>
        """

        return data, score_html, analysis_html, profile_html, gr.update(interactive=True)

    except Exception as e:
        err_msg = f"Error: {str(e)}"
        return None, f"<div>{err_msg}</div>", f"<div>{err_msg}</div>", f"<div>{err_msg}</div>", gr.update(interactive=False)

def generate_message_action(user_data_state, consult_text, api_key):
    """
    [문구 생성] 버튼 클릭 시 실행: LLM 호출
    """
    if not user_data_state:
        return "<div>먼저 분석을 실행해주세요.</div>", "분석 데이터 없음"
    
    if not api_key:
        return "<div>API Key를 입력해주세요.</div>", "API Key 누락"

    # LLM 호출
    message = generate_marketing_message(user_data_state, consult_text, api_key)
    
    # 에러 체크
    if "오류 발생" in message or "API Key" in message:
         chat_bubble_html = f"""
        <div class="phone-screen-content">
            <div class="chat-bubble-ai" style="color: red;">
                {message}
            </div>
        </div>
        """
         return chat_bubble_html, message

    # 스마트폰 화면 내 말풍선 HTML
    chat_bubble_html = f"""
    <div class="phone-screen-content">
        <div class="chat-timestamp">오늘 오후 2:30</div>
        <div class="chat-bubble-ai">
            <div class="chat-profile">
                <div class="chat-avatar">R</div>
                <div class="chat-name">Retention AI</div>
            </div>
            <div class="chat-text">
                {message.replace(chr(10), '<br>')}
            </div>
        </div>
    </div>
    """
    
    return chat_bubble_html, message 

def send_message_action():
    time.sleep(0.5)
    return "메시지가 성공적으로 전송되었습니다!"


# ==========================================
# 2. CSS 스타일 (대시보드 & 폰 목업)
# ==========================================
css = """
/* 폰트 및 기본 설정 */
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
body { font-family: 'Pretendard', sans-serif; background-color: #f3f4f6; }

/* 카드 공통 스타일 */
.dashboard-card {
    background: white;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    height: 100%;
    min-height: 280px;
    display: flex;
    flex-direction: column;
}
.card-title {
    font-size: 14px;
    color: #6b7280;
    font-weight: 600;
    margin-bottom: 16px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* 1. 고객 프로필 카드 */
.profile-header { display: flex; flex-direction: column; align-items: center; text-align: center; }
.avatar { 
    width: 80px; height: 80px; border-radius: 50%; background-color: #e5e7eb; 
    margin-bottom: 12px; display: flex; align-items: center; justify-content: center; font-size: 32px;
}
.user-name { font-size: 24px; font-weight: 700; color: #111827; }
.user-id { font-size: 14px; color: #9ca3af; margin-bottom: 20px; }
.stat-row { display: flex; justify-content: space-between; width: 100%; margin-top: 8px; padding: 8px 0; border-bottom: 1px solid #f3f4f6; }
.stat-label { color: #6b7280; font-size: 14px; }
.stat-value { color: #111827; font-weight: 600; font-size: 14px; }
.badge-gold { color: #b45309; background-color: #fffbeb; padding: 2px 6px; border-radius: 4px; font-size: 12px; }

/* 2. 스코어 카드 */
.churn-score-container { display: flex; align-items: baseline; gap: 8px; margin-bottom: 12px; }
.churn-score-text { font-size: 48px; font-weight: 800; line-height: 1; }
.churn-badge { padding: 4px 8px; border-radius: 9999px; font-size: 12px; font-weight: 600; }
.badge-critical { background-color: #fef2f2; color: #ef4444; }
.badge-warning { background-color: #fffbeb; color: #f59e0b; }
.badge-safe { background-color: #ecfdf5; color: #10b981; }

.progress-bg { width: 100%; height: 12px; background-color: #f3f4f6; border-radius: 9999px; overflow: hidden; margin-bottom: 8px; }
.progress-bar { height: 100%; border-radius: 9999px; transition: width 1s ease-in-out; }
.sub-text { font-size: 12px; color: #9ca3af; text-align: right; }

/* 3. AI 분석 카드 */
.empty-state { 
    display: flex; align-items: center; justify-content: center; height: 100%; color: #9ca3af; font-size: 14px; 
    border: 2px dashed #e5e7eb; border-radius: 12px; padding: 20px; text-align: center;
}
.factors-list { flex: 1; overflow-y: auto; max-height: 180px; }
.factor-item { display: flex; gap: 12px; padding: 12px; border-radius: 12px; margin-bottom: 8px; align-items: center; }
.bg-red-50 { background-color: #fef2f2; border: 1px solid #fee2e2; }
.text-red-700 { color: #b91c1c; }
.bg-yellow-50 { background-color: #fffbeb; border: 1px solid #fef3c7; }
.text-yellow-700 { color: #b45309; }
.factor-text { display: flex; flex-direction: column; }
.factor-label { font-size: 11px; opacity: 0.8; font-weight: 600; }
.factor-desc { font-size: 14px; font-weight: 600; }
.recommendation-box { margin-top: 10px; background: #eff6ff; color: #1e40af; padding: 12px; border-radius: 8px; font-size: 13px; line-height: 1.4; }

/* 4. 스마트폰 목업 */
.phone-frame {
    width: 320px;
    height: 580px;
    background: #fff;
    border: 12px solid #1f2937; /* 다크 그레이 프레임 */
    border-radius: 40px;
    margin: 0 auto;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
}
.phone-notch {
    position: absolute; top: 0; left: 50%; transform: translateX(-50%);
    width: 120px; height: 24px; background: #1f2937;
    border-bottom-left-radius: 16px; border-bottom-right-radius: 16px; z-index: 10;
}
.phone-screen {
    width: 100%; height: 100%; background: #f3f4f6;
    display: flex; flex-direction: column; padding-top: 40px;
}
.phone-header {
    padding: 0 16px 12px; border-bottom: 1px solid #e5e7eb; background: white;
    font-size: 14px; font-weight: 600; color: #374151; text-align: center;
}
.phone-body {
    flex: 1; padding: 16px; overflow-y: auto; display: flex; flex-direction: column;
}

/* 채팅 말풍선 스타일 */
.chat-timestamp { text-align: center; font-size: 11px; color: #9ca3af; margin-bottom: 16px; }
.chat-bubble-ai { display: flex; gap: 10px; align-items: flex-start; animation: popIn 0.3s ease-out; }
.chat-profile { display: flex; flex-direction: column; align-items: center; gap: 4px; }
.chat-avatar { width: 32px; height: 32px; background: #4f46e5; color: white; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 14px; }
.chat-name { font-size: 10px; color: #6b7280; }
.chat-text {
    background: white; padding: 12px 16px; border-radius: 0 16px 16px 16px;
    font-size: 14px; color: #374151; line-height: 1.5; box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    max-width: 220px;
}

@keyframes popIn {
    0% { opacity: 0; transform: translateY(10px); }
    100% { opacity: 1; transform: translateY(0); }
}
"""

# ==========================================
# 3. Gradio UI 구성
# ==========================================

with gr.Blocks(title="ChurnGuard AI") as demo:
    
    # 상태 저장소
    user_state = gr.State({})
    message_state = gr.State("")

    # 헤더
    with gr.Row():
        gr.Markdown("## 🛡️ 고객 분석 & 서비스 추천")

        # --- 컨트롤 영역 ---
    with gr.Row(variant="panel"):
        with gr.Column(scale=2):
            input_user_id = gr.Textbox(label="Customer ID", value="Test-002", placeholder="고객 ID 입력")
        with gr.Column(scale=2):
            input_consult_text = gr.Textbox(label="최근 상담/고민 (VOC)", value="요금이 비싸서 해지를 고민중입니다.", placeholder="상담 내용 입력")
        with gr.Column(scale=1):
            btn_analyze = gr.Button("🔍 분석 실행", variant="primary", size="lg")

    # --- [상단] 대시보드 영역 (3개 컬럼) ---
    with gr.Row(equal_height=True):
        
        # 1. 고객 정보 카드
        with gr.Column(scale=1):
            user_profile_html = gr.HTML(
                """
                <div class="dashboard-card">
                    <div class="profile-header">
                        <div class="avatar">👤</div>
                        <div class="user-name">-</div>
                        <input class="user-id">ID를 입력하세요</div>
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
                    <div class="empty-state">분석 대기 중...</div>
                </div>
                """
            )

        # 3. AI 분석 결과 카드
        with gr.Column(scale=1):
            ai_analysis_html = gr.HTML(
                """
                <div class="dashboard-card">
                    <div class="card-title">🟣 AI 분석 이탈 원인</div>
                    <div class="empty-state">'분석 실행' 버튼을 눌러<br>원인을 파악하세요.</div>
                </div>
                """
            )
            
    # API 키 입력 (숨김)
    with gr.Accordion("⚙️ 설정 (API Key)", open=False):
        input_api_key = gr.Textbox(label="OpenAI API Key", value=DEFAULT_API_KEY, type="password")

    # --- [하단] 액션 영역 (폰 시뮬레이터) ---
    gr.Markdown("### 📱 Marketing Action")
    
    with gr.Row():
        # 왼쪽: 컨트롤 패널
        with gr.Column(scale=1):
            gr.Markdown("#### 마케팅 실행 옵션")
            gr.Info("AI 분석 결과를 바탕으로 개인화된 메시지를 생성합니다.")
            
            btn_gen_msg = gr.Button("✨ 마케팅 문구 생성 (AI)", variant="primary")
            btn_send_msg = gr.Button("🚀 전송 하기", variant="secondary", interactive=False)
            
            send_status = gr.Textbox(label="전송 상태", interactive=False)

        # 오른쪽: 스마트폰 목업
        with gr.Column(scale=2):
            phone_display = gr.HTML(
                """
                <div class="phone-frame">
                    <div class="phone-notch"></div>
                    <div class="phone-screen">
                        <div class="phone-header">Retention AI Message Preview</div>
                        <div class="phone-body" id="phone-content">
                            <div style="text-align: center; color: #9ca3af; margin-top: 50%;">미리보기 대기 중...</div>
                        </div>
                    </div>
                </div>
                """
            )

    # ==========================================
    # 4. 이벤트 연결
    # ==========================================

    # 1) 분석 버튼 클릭
    btn_analyze.click(
        fn=analyze_customer,
        inputs=[input_user_id, input_consult_text],
        outputs=[user_state, churn_score_html, ai_analysis_html, user_profile_html, btn_send_msg]
    )

    # 2) 문구 생성 버튼 클릭
    btn_gen_msg.click(
        fn=generate_message_action,
        inputs=[user_state, input_consult_text, input_api_key],
        outputs=[phone_display, message_state]
    ).then(
        lambda: gr.update(interactive=True), None, [btn_send_msg]
    )

    # 3) 전송 버튼 클릭
    btn_send_msg.click(
        fn=send_message_action,
        inputs=None,
        outputs=[send_status]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=css)