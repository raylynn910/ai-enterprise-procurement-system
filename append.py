with open('backend/main.py', 'a', encoding='utf-8') as f:
    f.write('''
@app.get('/api/recommend/suppliers')
def recommend_suppliers(category: str, scenario: str):
    """Supplier Recommendation API"""
    conn = sqlite3.connect('procurement.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    order_by_clause = ''
    if scenario == 'cost':
        order_by_clause = 'ORDER BY Avg_Savings DESC'
    elif scenario == 'urgent':
        order_by_clause = 'ORDER BY Avg_Days_Late ASC'
    elif scenario == 'compliance':
        order_by_clause = 'ORDER BY Avg_ESG DESC'
    else:
        order_by_clause = 'ORDER BY Avg_Savings DESC'
    
    cursor.execute(f"""
        SELECT Supplier_Name, Supplier_Country, 
               AVG(Savings_Pct) as Avg_Savings, 
               AVG(Days_Late) as Avg_Days_Late, 
               AVG(Supplier_ESG_Score) as Avg_ESG
        FROM procurement_data 
        WHERE Category = ? 
        GROUP BY Supplier_Name 
        {order_by_clause} 
        LIMIT 3
    """, (category,))
    
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for i, row in enumerate(rows):
        score_str = ''
        if scenario == 'cost':
            score_str = f"{row['Avg_Savings']:.2f}% Savings"
            reason = f"Historical high cost reduction in {category}"
        elif scenario == 'urgent':
            score_str = f"{row['Avg_Days_Late']:.1f} Days Late"
            reason = f"Reliable and fast delivery track record"
        elif scenario == 'compliance':
            score_str = f"ESG: {row['Avg_ESG']:.1f}"
            reason = f"Strong adherence to sustainability and compliance"
        
        results.append({
            'rank': i + 1,
            'name': row['Supplier_Name'],
            'country': row['Supplier_Country'],
            'score_text': score_str,
            'reason': reason
        })
    return results
''')
