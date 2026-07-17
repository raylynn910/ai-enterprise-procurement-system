with open('backend/main.py', 'a', encoding='utf-8') as f:
    f.write('''
@app.get('/api/overview/kpis')
def overview_kpis():
    """Overview KPIs API"""
    conn = sqlite3.connect('procurement.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # 1. Avg Savings, Avg ESG
    cursor.execute('SELECT AVG(Savings_Pct) as avg_savings, AVG(Supplier_ESG_Score) as avg_esg FROM procurement_data')
    row1 = cursor.fetchone()
    avg_savings = float(row1['avg_savings']) if row1['avg_savings'] is not None else 0.0
    avg_esg = float(row1['avg_esg']) if row1['avg_esg'] is not None else 0.0
    
    # 2. Risk Score & High Risk Count
    cursor.execute('SELECT Supplier_Risk, COUNT(*) as cnt FROM procurement_data GROUP BY Supplier_Risk')
    risk_rows = cursor.fetchall()
    total_risk_score = 0
    total_count = 0
    high_risk_count = 0
    for r in risk_rows:
        cnt = int(r['cnt'])
        risk_val = str(r['Supplier_Risk']).strip().capitalize()
        total_count += cnt
        if risk_val == 'High':
            total_risk_score += 100 * cnt
            high_risk_count += cnt
        elif risk_val == 'Medium':
            total_risk_score += 50 * cnt
            
    avg_risk_score = (total_risk_score / total_count) if total_count > 0 else 0
    
    # 3. Maverick Spend Count
    cursor.execute("SELECT COUNT(*) as cnt FROM procurement_data WHERE Maverick_Spend COLLATE NOCASE IN ('yes', 'true', '1')")
    maverick_row = cursor.fetchone()
    maverick_count = int(maverick_row['cnt'])
    
    conn.close()
    
    return {
        'avg_savings': round(avg_savings, 2),
        'avg_risk_score': round(avg_risk_score, 1),
        'high_risk_count': high_risk_count,
        'avg_esg': round(avg_esg, 1),
        'maverick_count': maverick_count
    }
''')
