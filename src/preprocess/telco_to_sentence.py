import pandas as pd

def telco_to_sentence(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    def yes_no(val):
        return "이용" if val == 1 or val == "Yes" else "미이용"

    sentences = []

    for _, row in df.iterrows():

        # ===== 1. 기본 정보 =====
        gender = "남성" if row["Gender"] == "Male" else "여성"
        married = "기혼" if row["Married"] == "Yes" else "미혼"
        dependents = "부양가족 있음" if row["Dependents"] == 1 else "부양가족 없음"

        # AgeGroup + Age
        age_group = row["AgeGroup"] if pd.notnull(row["AgeGroup"]) else ""
        age_info = f"{age_group} {int(row['Age'])}세"

        # ===== 2. 서비스 사용 여부 =====
        online_security = yes_no(row["OnlineSecurity"])
        online_backup = yes_no(row["OnlineBackup"])
        tech_support = yes_no(row["TechSupport"])
        unlimited_data = yes_no(row["UnlimitedData"])

        # ===== 3. 청구 & 결제 방법 =====
        paperless = (
            "종이 없는 청구서 사용"
            if row["PaperlessBilling"] in [1, "Yes"]
            else "종이 청구서 사용"
        )
        payment = row["PaymentMethod"]

        # ===== 4. 사용 정보 =====
        download_gb = row["AvgDownloadGB"]
        monthly_charge = row["Monthly_charge"]
        total_roam = row["TotalRoamCharge"]
        tenure = row["Tenure_month"]
        ltv = row["CustomerLTV"]
        satis = row["SatisScore"]

        # ===== 5. 최종 문장 구성 =====
        sentence = (
            f"이 고객은 {age_info} {gender}이며 {married}, {dependents}이다. "
            f"온라인 보안 {online_security}, 온라인 백업 {online_backup}, 기술 지원 {tech_support}, "
            f"무제한 데이터 {unlimited_data} 서비스를 이용한다. "
            f"월 평균 다운로드량은 {download_gb}GB이며 월 요금은 {monthly_charge}달러이다. "
            f"로밍 요금은 총 {total_roam}달러이며 서비스 이용 기간은 {tenure}개월이다. "
            # f"고객 생애가치는 {ltv}이며 만족도는 {satis}점이다. "
            f"{paperless}하며 결제 방식은 {payment}이다."
        )

        sentences.append(sentence)

    # df에 text 컬럼 추가
    df["text"] = sentences

    return df


if __name__ == "__main__":
    input_path = "data/processed/telco_cleaned_data.csv"
    df_with_text = telco_to_sentence(input_path)
    print(df_with_text[["CustomerId", "text"]].head())
