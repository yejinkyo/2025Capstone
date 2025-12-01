# src/msg_generator.py
import os
from openai import OpenAI
# 같은 폴더 내 prompts.py에서 가져옴
from src.llm.prompts import SYSTEM_PROMPT, format_user_prompt

def generate_marketing_message(analysis_json: dict, consult_text: str, api_key: str):
    """
    OpenAI API를 호출하여 마케팅 메세지 생성
    Args:
        analysis_json (dict): data_generator.py에서 생성한 분석 결과 JSON
        consult_text (str): 유저 상담 텍스트
        api_key (str): OpenAI API Key
    """
    
    if not api_key:
        return "⚠️ OpenAI API Key가 입력되지 않았습니다. 설정 탭에서 키를 입력해주세요."
    
    if not analysis_json or "error" in analysis_json:
        return "⚠️ 분석 데이터가 올바르지 않습니다. 먼저 [추천 로직 실행]을 완료해주세요."

    try:
        # 1. 프롬프트 생성
        user_prompt_text = format_user_prompt(analysis_json, consult_text)
        
        # 2. OpenAI 클라이언트 초기화
        client = OpenAI(api_key=api_key)

        # 3. API 호출
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # 혹은 gpt-4o
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt_text}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        return response.choices[0].message.content

    except Exception as e:
        return f"LLM 생성 중 오류 발생:\n{str(e)}"