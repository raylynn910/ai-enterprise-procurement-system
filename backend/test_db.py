import sqlite3
import pandas as pd

conn = sqlite3.connect('procurement.db')
df = pd.read_sql_query("SELECT * FROM procurement_data WHERE Supplier_ID LIKE '%中聯油脂%'", conn)
print(df)
conn.close()
