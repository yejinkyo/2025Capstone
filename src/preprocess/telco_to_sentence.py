import pandas as pd

def telco_to_sentence(input_path: str) -> pd.DataFrame:
    """
    텔코 데이터를 문장형 텍스트로 변환하는 함수
    """
    # 1. 데이터 불러오기
    df = pd.read_csv(input_path, encoding='utf-8')

    # 2. 문장형 텍스트 생성
    def generate_sentence(row):
        # 기본 정보
        gender = "남성" if row ["Gender"] == "Male" else "여성"
        age = int(row["Age"])
        age_group = "청년" if age < 30 else "중년" if age < 60 else "노년"
        married = "기혼" if row["Married"] == "Yes" else "미혼"
        dependents = "부양가족 있음" if row["Dependents"] == "Yes" else "부양가족 없음"

        # 계약 및 결제 정보
        billing = "전자 청구서" if row["PaperlessBilling"] == "Yes" else "종이 청구서"
        payment = row["PaymentMethod"]
        unlimited_data = "무제한 데이터 사용" if row["UnlimitedData"] == "Yes" else "제한된 데이터 사용"
        avg_download = row["AvgDownloadGB"]

        # 서비스 정보
        online_security = "온라인 보안 서비스 이용" if row["OnlineSecurity"] == "Yes" else "온라인 보안 서비스 미이용"
        online_backup = "온라인 백업 서비스 이용" if row["OnlineBackup"] == "Yes" else "온라인 백업 서비스 미이용"
        tech_support = "기술 지원 서비스 이용" if row["TechSupport"] == "Yes" else "기술 지원 서비스 미이용"

        # 서비스 이용
        # tenure_months = row["ServiceDuration"]

        # 이탈 여부 및 이유
        churn = "이탈했습니다" if row["ChurnLabel"] == "Yes" else "유지하고 있습니다"
        reason = row["ChurnReason"] if pd.notna(row["ChurnReason"]) else "없음"

        # 최종 문장 생성
        sentence = (
            f"{age}세({age_group})의 {gender} 고객, {married}, {dependents}입니다. "
            f"청구 방식은 {billing}이며, 결제 방식은 {payment}입니다. "
            f"{unlimited_data}, 평균 다운로드 {avg_download}GB입니다. "
            f"{online_security}, {online_backup}, {tech_support}. "
            # f"서비스 이용 기간은 {tenure_months}일입니다. "
            f"고객은 서비스를 {churn}. 해지 이유: {reason}."
        )
        return sentence
    
    df["text"] = df.apply(generate_sentence, axis=1)

    return df


if __name__ == "__main__":
    input_path = "data/processed/telco_cleaned_data.csv"
    df_with_text = telco_to_sentence(input_path)
    print(df_with_text[["CustomerId", "text"]].head())