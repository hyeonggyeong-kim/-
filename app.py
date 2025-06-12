import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정 (윈도우용 예시)
font_path = "C:/Windows/Fonts/malgun.ttf"
font_prop = fm.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_prop)
plt.rcParams['axes.unicode_minus'] = False

# 모델 로드
@st.cache_resource
def load_model():
    with open("sample_ddos_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model, model.feature_names_in_.tolist()

model, required_features = load_model()

def prepare_input(df):
    df.columns = df.columns.str.strip()
    for col in required_features:
        if col not in df.columns:
            df[col] = 0
    return df[required_features].replace([np.inf, -np.inf], np.nan).fillna(0)

st.title("🧠 DDoS 탐지 시스템 (고신뢰 탐지만 저장 + 이상 분포 시각화)")

# ✅ 이 부분만 경로 변경!
folder_path = st.text_input(
    "📂 분석할 CSV 폴더 경로:", r"C:\Users\gudru\.vscode\sw\MachineLearningCVE"
)
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

if os.path.isdir(folder_path):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    if st.button("🚀 전체 분석 시작"):
        for file_name in csv_files:
            file_path = os.path.join(folder_path, file_name)
            try:
                df = pd.read_csv(file_path)
                df_input = prepare_input(df)

                probas = model.predict_proba(df_input)
                high_conf_ddos = probas[:, 1] > 0.99
                attack_count = np.sum(high_conf_ddos)

                st.subheader(f"📁 파일명: {file_name}")
                if attack_count >= 1500:
                    st.error(f"🚨 DDoS 의심 행 {attack_count}건 탐지됨! (확률 > 99%) → 저장됨")

                    result_df = df.copy()
                    result_df["DDoS_Prob"] = probas[:, 1]
                    result_df[high_conf_ddos].to_csv(
                        os.path.join(output_dir, f"ddos_detected_{file_name}"), index=False
                    )
                elif attack_count > 0:
                    st.warning(f"⚠️ 낮은 확률 의심 행 {attack_count}건 존재 (저장 안 됨)")
                else:
                    st.success("✅ DDoS 의심 없음 (정상 트래픽)")

                # 이상 확률 > 99% 행 히스토그램 시각화
                if attack_count > 0:
                    high_prob_indices = np.where(high_conf_ddos)[0]
                    segment_size = 1000
                    max_index = len(df_input)
                    bins = np.arange(0, max_index + segment_size, segment_size)
                    hist, bin_edges = np.histogram(high_prob_indices, bins=bins)

                    fig, ax = plt.subplots(figsize=(10, 2.5))
                    ax.bar(bin_edges[:-1], hist, width=segment_size, align='edge', color='red', alpha=0.7)
                    ax.set_title(f"{file_name} - 구간별 DDoS 의심 트래픽 수")
                    ax.set_xlabel("샘플 인덱스 (구간별)")
                    ax.set_ylabel("이상 행 개수")
                    st.pyplot(fig)

            except Exception as e:
                st.warning(f"⚠️ {file_name} 처리 중 오류 발생: {e}")
else:
    st.warning("📁 경로가 잘못되었거나 폴더가 존재하지 않음.")