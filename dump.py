import sqlite3
conn = sqlite3.connect('backend/procurement.db')
for row in conn.execute("SELECT sql FROM sqlite_master WHERE type='table'"):
    if row[0]: print(row[0])
