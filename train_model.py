import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ✅ 1. CSV 경로
benign_path = r"C:\Users\gudru\.vscode\sw\MachineLearningCVE\Monday-WorkingHours.pcap_ISCX.csv"
ddos_path = r"C:\Users\gudru\.vscode\sw\MachineLearningCVE\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

# ✅ 2. CSV 로딩
print("📥 CSV 로딩 중...")
df_benign = pd.read_csv(benign_path)
print("✅ BENIGN 로드 완료:", df_benign.shape)

df_ddos = pd.read_csv(ddos_path)
print("✅ DDoS 로드 완료:", df_ddos.shape)

# ✅ 3. 컬럼 공백 제거
df_benign.columns = df_benign.columns.str.strip()
df_ddos.columns = df_ddos.columns.str.strip()
print("✅ 컬럼 정리 완료")

# ✅ 4. 라벨 부여
df_benign["Label"] = 0
df_ddos["Label"] = 1
print("✅ 라벨 부여 완료")

# ✅ 5. 병합
df = pd.concat([df_benign, df_ddos], ignore_index=True)
print("✅ 병합 완료:", df.shape)

# ✅ 6. 사용할 피처 지정
selected_features = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std'
]

print("✅ 피처 추출 중...")
df = df[selected_features + ['Label']].replace([np.inf, -np.inf], np.nan).dropna()
print("✅ 피처 준비 완료:", df.shape)

X = df[selected_features]
y = df['Label']

# ✅ 7. 학습/테스트 분할
print("✅ 데이터셋 분할 중...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print("✅ 분할 완료:", X_train.shape, X_test.shape)

# ✅ 8. 모델 학습
print("🧠 모델 학습 중...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("✅ 모델 학습 완료")

# ✅ 9. 성능 평가
y_pred = model.predict(X_test)
print("\n📊 분류 리포트:\n")
print(classification_report(y_test, y_pred, target_names=['BENIGN', 'DDoS']))

# ✅ 10. 저장
print("💾 모델 저장 중...")
with open("sample_ddos_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("🎉 학습 완료! 'sample_ddos_model.pkl'로 저장됨 ✅")