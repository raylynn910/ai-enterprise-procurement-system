import csv
import sqlite3
import os

# 路徑設定
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, 'dataset', 'data_final.csv')
DB_PATH = os.path.join(BASE_DIR, 'backend', 'procurement.db')
TABLE_NAME = 'procurement_data'

def import_csv_to_sqlite():
    print(f"開始讀取 CSV: {CSV_PATH}")
    if not os.path.exists(CSV_PATH):
        print("錯誤：找不到 CSV 檔案。")
        return

    # 連線到 SQLite (若檔案不存在會自動建立)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    with open(CSV_PATH, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        headers = next(reader)
        
        # 處理欄位名稱 (將空格替換為底線，並過濾掉特殊字元)
        clean_headers = [h.strip().replace(' ', '_').replace('-', '_') for h in headers]
        
        # 動態建立 CREATE TABLE 語句 (全部預設為 TEXT 以求彈性，後續可依需求改變)
        columns_sql = ", ".join([f'"{col}" TEXT' for col in clean_headers])
        create_table_sql = f"CREATE TABLE IF NOT EXISTS {TABLE_NAME} ({columns_sql});"
        
        print(f"正在建立資料表 {TABLE_NAME}...")
        cursor.execute(f"DROP TABLE IF EXISTS {TABLE_NAME};") # 每次執行先清空舊資料
        cursor.execute(create_table_sql)
        
        # 動態建立 INSERT 語句
        placeholders = ", ".join(["?"] * len(clean_headers))
        insert_sql = f"INSERT INTO {TABLE_NAME} VALUES ({placeholders})"
        
        # 批次新增資料
        print("正在匯入資料...")
        rows = [row for row in reader]
        cursor.executemany(insert_sql, rows)
        
        conn.commit()
        print(f"成功！已將 {len(rows)} 筆資料匯入至 {DB_PATH} 內的 {TABLE_NAME} 表格中。")
        
    conn.close()

if __name__ == '__main__':
    import_csv_to_sqlite()
