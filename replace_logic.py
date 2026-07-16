import re

with open('backend/main.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = '''        # --- Calculate 5 Radar Chart Dimensions ---
        c_score = 0.0 if is_guardrail_blocked else float(esg_score)
        f_score = max(10.0, 100.0 - risk_score)
        
        # Delivery Score based on days late
        d_late = float(days_late) if days_late > 0 else 0.0
        d_score = max(10.0, 100.0 - (d_late * 3.5)) # 0 days -> 100, 20 days -> 30
        
        if 'auth_esg' in locals() and auth_esg:
            esg_score = auth_esg['esg_score']
            c_score = auth_esg['compliance']
            d_score = auth_esg['delivery_score']
            
        # --- Real Financial OSINT API ---'''

replacement = '''        # --- Calculate 5 Radar Chart Dimensions ---
        rep_score = 50.0  # Base reputation score
        
        # Adjust reputation based on OSINT volume
        if len(osint_sources_list) >= 3:
            rep_score += 20.0
        elif len(osint_sources_list) >= 1:
            rep_score += 10.0
            
        # Adjust reputation based on Sentiment
        if is_guardrail_blocked:
            rep_score = 10.0
        elif "正面" in osint_summary_msg or "優良" in osint_summary_msg or "獲獎" in osint_summary_msg:
            rep_score += 15.0
            
        rep_score = min(100.0, max(0.0, rep_score))
        
        f_score = max(10.0, 100.0 - risk_score)
        
        # Delivery Score based on days late
        d_late = float(days_late) if days_late > 0 else 0.0
        d_score = max(10.0, 100.0 - (d_late * 3.5)) # 0 days -> 100, 20 days -> 30
        
        if 'auth_esg' in locals() and auth_esg:
            esg_score = auth_esg['esg_score']
            rep_score = max(rep_score, 85.0)  # Authoritative source implies high reputation
            d_score = auth_esg['delivery_score']
            
        # --- Real Financial OSINT API ---'''

if target in content:
    content = content.replace(target, replacement)
else:
    print("Target block 1 not found!")

target2 = '''            is_mock=False,
            compliance_score=round(c_score, 1),
            financial_score=round(f_score, 1),
            delivery_score=round(d_score, 1),
            esg_score=round(float(esg_score), 1),
            pricing_score=round(p_score, 1),'''

replacement2 = '''            is_mock=False,
            reputation_score=round(rep_score, 1),
            financial_score=round(f_score, 1),
            delivery_score=round(d_score, 1),
            esg_score=round(float(esg_score), 1),
            pricing_score=round(p_score, 1),'''

if target2 in content:
    content = content.replace(target2, replacement2)
else:
    print("Target block 2 not found!")

with open('backend/main.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Replaced successfully!")
