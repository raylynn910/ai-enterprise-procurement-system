import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import pickle
import os

print("--- 啟動第一組：供應商風險預測模型訓練 ---")

# Load the CSV file into a DataFrame
df = pd.read_csv('Data_中英欄位.csv')
print(f"資料表大小: {df.shape}")

# 移除中文字元與括號，確保欄位錨定 (與 Team 2 作法一致)
df.columns = [col.split("（")[0].strip() for col in df.columns]

# Define the target variable y
y = df['Supplier Risk']

# Define the features X
X_selected = df[['Supplier ESG Score', 'On Time Delivery', 'Days Late', 'PO Status']]

# Handle categorical features in X_selected by one-hot encoding 'PO Status'
categorical_cols_selected = X_selected.select_dtypes(include='object').columns

if not categorical_cols_selected.empty:
    print(f"以下是類別變數: {list(categorical_cols_selected)}")
    X_selected = pd.get_dummies(X_selected, columns=categorical_cols_selected, drop_first=True)
else:
    print("不需要進行類別處理。")

# 記下訓練時的欄位順序，以便預測時對齊
training_columns = X_selected.columns.tolist()

# Perform the train-test split (70% training, 30% testing)
X_train_selected, X_test_selected, y_train_selected, y_test_selected = train_test_split(
    X_selected, y, test_size=0.3, random_state=1, stratify=y
)

# Scaling
scaler_selected = StandardScaler()
X_train_selected_scaled = scaler_selected.fit_transform(X_train_selected)
X_test_selected_scaled = scaler_selected.transform(X_test_selected)

# Train Model
rf_model = RandomForestClassifier(
    n_estimators=200,      
    max_depth=10,          
    min_samples_split=5,   
    class_weight='balanced_subsample', 
    random_state=42
)
rf_model.fit(X_train_selected_scaled, y_train_selected)

# Predict and Evaluate
y_pred_selected = rf_model.predict(X_test_selected_scaled)
print("\n--- 模型效能 ---")
accuracy_selected = accuracy_score(y_test_selected, y_pred_selected)
print(f"準確率 (Accuracy): {accuracy_selected:.4f}")
print("\n分類報告 (Classification Report):\n")
print(classification_report(y_test_selected, y_pred_selected))

# Export Models
print("\n[Exporting Models] 開始打包模型...")
os.makedirs("../backend/models", exist_ok=True)

with open("../backend/models/risk_rf_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)
    
with open("../backend/models/risk_scaler.pkl", "wb") as f:
    pickle.dump(scaler_selected, f)
    
with open("../backend/models/risk_features.pkl", "wb") as f:
    pickle.dump(training_columns, f)

print("[Success] 模型已成功匯出至 backend/models/ ! 包含: risk_rf_model.pkl, risk_scaler.pkl, risk_features.pkl")
