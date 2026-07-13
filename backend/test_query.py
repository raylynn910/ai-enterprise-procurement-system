import sqlite3
conn = sqlite3.connect('procurement.db')
cursor = conn.cursor()

print("==== High Risk Suppliers ====")
cursor.execute("SELECT Supplier_ID, Supplier_Name, Supplier_Risk, Supplier_ESG_Score, Days_Late FROM procurement_data WHERE Supplier_Risk='High' GROUP BY Supplier_ID LIMIT 3")
for row in cursor.fetchall():
    print(row)

print("==== Medium Risk Suppliers ====")
cursor.execute("SELECT Supplier_ID, Supplier_Name, Supplier_Risk, Supplier_ESG_Score, Days_Late FROM procurement_data WHERE Supplier_Risk='Medium' GROUP BY Supplier_ID LIMIT 3")
for row in cursor.fetchall():
    print(row)

print("==== Low Risk Suppliers ====")
cursor.execute("SELECT Supplier_ID, Supplier_Name, Supplier_Risk, Supplier_ESG_Score, Days_Late FROM procurement_data WHERE Supplier_Risk='Low' GROUP BY Supplier_ID LIMIT 3")
for row in cursor.fetchall():
    print(row)

conn.close()
