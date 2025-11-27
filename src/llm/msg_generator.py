import os
from openai import OpenAI
from prompts import SYSTEM_PROMPT, format_user_prompt
import json
from dotenv import load_dotenv

load_dotenv()

# ===========================================
# 1. 고객 분석 결과 데이터 조회
# ===========================================
def get_customer_analysis(user_id, data_path="data/user_analysis.json"):
    """
    JSON 파일에서 user id에 해당하는 분석 정보를 읽어 반환
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    return all_data.get(str(user_id))

# ===========================================
# 2. GPT API 호출
# ===========================================
def generate_marketing_message(user_id, api_key):
    """
    OpenAI API를 호출하여 마케팅 메세지 생성
    """
    # 1. 데이터 조회 
    data = get_customer_analysis(user_id)

    # 프롬프트 텍스트 병합
    user_prompt_text = format_user_prompt(data)

    print(f"=== GPT 요청 ===")

    # 3. OpenAI API 호출
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt_text}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    
    return response.choices[0].message.content


if __name__ == '__main__':
    MY_API_KEY = os.getenv("OPENAI_API_KEY")

    target_user_list = ["10001", "10002"]

    for user_id in target_user_list:
        result = generate_marketing_message(user_id, MY_API_KEY)

        print(result)