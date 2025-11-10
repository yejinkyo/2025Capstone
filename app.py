import streamlit as st
import pandas as pd
from src.model.rec import generate_recommendations

st.set_page_config(page_title="텔코 고객 추천 서비스", layout="wide")

st.title("텔코 고객 추천 서비스 🚀")
st.markdown("KMeans 군집 기반 + 기존 추천 로직 + 협업필터링 결과를 확인할 수 있습니다.")

# 1️⃣ 데이터 불러오기
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/telco_cleaned_data.csv", encoding="utf-8")
    df = generate_recommendations(df)
    return df

df = load_data()

# 2️⃣ 고객 선택
customer_list = df['CustomerId'].tolist()
selected_customer = st.selectbox("추천 결과를 보고 싶은 고객을 선택하세요:", customer_list)

customer_data = df[df['CustomerId'] == selected_customer].iloc[0]

# 3️⃣ 고객 정보 및 추천 결과 표시
st.subheader("고객 정보")
st.write({
    "CustomerId": customer_data['CustomerId'],
    "Cluster": customer_data['cluster_name'],
    "Satisfaction Score": customer_data['SatisScore'],
    "Churn Score": customer_data['ChurnScore']
})

st.subheader("추천 서비스")
st.write(customer_data['RecommendedServices'])

# 4️⃣ 전체 데이터 확인
if st.checkbox("전체 추천 결과 확인"):
    st.dataframe(df[['CustomerId', 'cluster_name', 'SatisScore', 'ChurnScore', 'RecommendedServices']])

