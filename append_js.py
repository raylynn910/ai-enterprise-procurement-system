with open('smart-procurement-ui/dashboard.js', 'a', encoding='utf-8') as f:
    f.write('''

// --- Dynamic Overview KPIs ---
async function loadOverviewKPIs() {
    try {
        const response = await fetch('http://localhost:8000/api/overview/kpis');
        if (!response.ok) throw new Error('API fetch failed');
        const data = await response.json();

        // 1. Avg Savings
        const tagSavings = document.getElementById('tag-est-savings');
        if (document.getElementById('val-est-savings')) {
            document.getElementById('val-est-savings').innerHTML = data.avg_savings + '% <span id="tag-est-savings" class="trend up"><i class="ph ph-trend-up"></i></span>';
        }

        // 2. Avg Risk
        const elRisk = document.getElementById('val-avg-risk');
        if (elRisk) {
            let riskClass = 'success';
            let riskText = '安全';
            if (data.avg_risk_score > 60) { riskClass = 'danger'; riskText = '高危險'; }
            else if (data.avg_risk_score > 30) { riskClass = 'warning'; riskText = '中等'; }
            elRisk.innerHTML = data.avg_risk_score + ' / 100 <span id="tag-avg-risk" class="status-tag ' + riskClass + '">' + riskText + '</span>';
        }

        // 3. High Risk Count
        const elHighRisk = document.getElementById('val-high-risk-count');
        if (elHighRisk) {
            let hrClass = data.high_risk_count > 10 ? 'danger' : 'success';
            let hrText = data.high_risk_count > 10 ? '需關注' : '健康';
            elHighRisk.innerHTML = data.high_risk_count + ' 筆 <span id="tag-high-risk-count" class="status-tag ' + hrClass + '">' + hrText + '</span>';
        }

        // 4. Avg ESG
        const elEsg = document.getElementById('val-avg-esg');
        if (elEsg) {
            let esgClass = data.avg_esg > 70 ? 'success' : 'warning';
            let esgText = data.avg_esg > 70 ? '健康' : '待加強';
            elEsg.innerHTML = data.avg_esg + ' / 100 <span id="tag-avg-esg" class="status-tag ' + esgClass + '">' + esgText + '</span>';
        }

        // 5. Maverick Count
        const elMav = document.getElementById('val-maverick-count');
        if (elMav) {
            let mavClass = data.maverick_count > 0 ? 'danger' : 'success';
            let mavText = data.maverick_count > 0 ? '需關注' : '合規';
            elMav.innerHTML = data.maverick_count + ' 筆 <span id="tag-maverick-count" class="status-tag ' + mavClass + '">' + mavText + '</span>';
        }
    } catch (e) {
        console.error('Failed to load KPIs:', e);
    }
}
''')
