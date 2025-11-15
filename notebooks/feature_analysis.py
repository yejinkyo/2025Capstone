import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_corr(df, output_path):
    """
    상관관계 히트맵

    Parameters
    -----------
    df: pandas.DataFrame
        분석할 df
    output_path: str
        히트맵 이미지 저장 경로
    """
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    corr = numeric_df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return corr


if __name__ == "__main__":
    df = pd.read_csv("data/processed/telco_cleaned_data.csv", encoding='utf-8')
    corr_matrix = plot_corr(df, "results/feature_corrheatmap.png")
    print(corr_matrix)