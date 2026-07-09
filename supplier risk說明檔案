這份 Notebook 主要用於執行供應商風險預測（Supplier Risk Prediction）的機器學習流程。
它使用了 Python 的 scikit-learn 函式庫，透過隨機森林（Random Forest）模型來預測供應商的風險等級。

以下是該 Notebook 的運作流程說明：
1. 資料載入與準備資料來源：從 Google Drive 載入 CSV 檔案。特徵選擇：選取了影響風險的關鍵特徵，包括：供應商 ESG 評分 (Supplier ESG Score)是否準時交貨 (On Time Delivery)延誤天數 (Days Late)採購單狀態 (PO Status)資料預處理：獨熱編碼 (One-Hot Encoding)：將類別變數（如「是否準時交貨」、「採購單狀態」）轉換為機器學習模型可讀取的數值格式。

資料分割：將資料集分為訓練集（70%）與測試集（30%），並使用 stratify=y 確保訓練與測試資料中各類別的比例維持一致。

2. 模型建置與訓練特徵縮放 (Scaling)：使用 StandardScaler 對特徵進行標準化，讓各數值變數處於相同的尺度，提升模型訓練穩定性。

模型選擇：採用 隨機森林分類器 (RandomForestClassifier)。

參數調優：設定 n_estimators=200（增加樹的數量提升準確度）、max_depth=10（限制深度防止過擬合）、class_weight='balanced_subsample'（處理類別不平衡問題）。

3. 模型評估模型在測試集上的表現相當出色，準確度（Accuracy）達到 0.9731，具體指標如下：指標 (Metric)結果 (Result)準確度 (Accuracy)0.97精確度 (Precision)0.97召回率 (Recall)0.97F1-score0.97混淆矩陣分析從混淆矩陣的結果可以看出，該模型在預測「High（高風險）」、「Low（低風險）」與「Medium（中風險）」時都有很高的判斷準確度，特別是在高風險與低風險類別上，幾乎沒有誤判。
