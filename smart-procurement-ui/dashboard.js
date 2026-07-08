// dashboard.js

document.addEventListener('DOMContentLoaded', () => {
    // SPA Navigation Logic
    const navItems = document.querySelectorAll('.nav-item[data-target]');
    const views = document.querySelectorAll('.view-section');

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            
            // Remove active class from all nav items
            navItems.forEach(nav => nav.classList.remove('active'));
            // Add active class to clicked item
            item.classList.add('active');

            // Hide all views
            views.forEach(view => {
                view.classList.remove('active');
                view.style.display = 'none';
            });

            // Show target view
            const targetId = item.getAttribute('data-target');
            const targetView = document.getElementById(targetId);
            if (targetView) {
                targetView.style.display = 'block';
                // Small timeout to allow display:block to apply before animating opacity
                setTimeout(() => {
                    targetView.classList.add('active');
                }, 10);
            }
        });
    });

    // Prediction Form Logic
    const btnPredict = document.getElementById('btn-predict');
    const predictionResult = document.getElementById('prediction-result');
    const valPredSavings = document.getElementById('val-pred-savings');
    const valPredAmount = document.getElementById('val-pred-amount');
    const predictionResult = document.getElementById('prediction-result');
    
    if (btnPredict) {
        btnPredict.addEventListener('click', async () => {
            // Add simple loading state
            btnPredict.innerHTML = '<i class="ph ph-spinner-gap ph-spin"></i> 預測中...';
            btnPredict.disabled = true;

            const supplier = document.getElementById('input-supplier').value;
            const category = document.getElementById('input-category') ? document.getElementById('input-category').value : 'IT Software';
            const budget = parseFloat(document.getElementById('input-budget').value) || 0;
            
            try {
                // Call POST /api/predict/supplier-risk API
                const response = await fetch('http://localhost:8000/api/predict/supplier-risk', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        supplier_id: supplier || 'SUP-001',
                        category: category
                    })
                });
                
                const result = await response.json();
                
                // For demonstration, map risk score back to savings for the original UI (or just display the risk output)
                // The UI expects Savings Pct and Savings Amount.
                // We'll calculate fake savings based on risk_score (lower risk = higher savings)
                let fakeSavingsPct = 10 - (result.risk_score / 10);
                let fakeAmount = (budget * fakeSavingsPct) / 100;
                
                // Update UI
                if (fakeSavingsPct > 0) {
                    valPredSavings.innerText = '+' + fakeSavingsPct.toFixed(1) + '%';
                    valPredSavings.className = 'result-value success-text';
                } else {
                    valPredSavings.innerText = fakeSavingsPct.toFixed(1) + '%';
                    valPredSavings.className = 'result-value danger-text';
                }
                
                valPredAmount.innerText = '$' + fakeAmount.toFixed(2);
                
                // Show recommendation text in a small alert below the amount
                let recDiv = document.getElementById('pred-recommendation');
                if (!recDiv) {
                    recDiv = document.createElement('div');
                    recDiv.id = 'pred-recommendation';
                    recDiv.style.marginTop = '10px';
                    recDiv.style.fontSize = '0.9rem';
                    predictionResult.appendChild(recDiv);
                }
                recDiv.innerHTML = `<strong class="${result.risk_level === 'High' ? 'danger-text' : (result.risk_level === 'Medium' ? 'warning-text' : 'success-text')}">風險等級: ${result.risk_level}</strong><br/>${result.recommendation}`;
                
                // Show result panel
                predictionResult.style.display = 'block';
                
            } catch (error) {
                console.error('API Error:', error);
                alert('API 呼叫失敗，請確認 FastAPI 後端已啟動。');
            } finally {
                // Reset button
                btnPredict.innerHTML = '<i class="ph ph-magic-wand"></i> 執行預測';
                btnPredict.disabled = false;
            }
        });
    }

    // Reports Button Logic
    document.getElementById('btn-weekly-report')?.addEventListener('click', () => {
        alert('報表生成中... (此處可串接後端 API 下載 PDF)');
    });
    document.getElementById('btn-monthly-report')?.addEventListener('click', () => {
        alert('報表生成中... (此處可串接後端 API 下載 Excel/PDF)');
    });
});

// Global suppliers array populated from API
let mockSuppliers = [];

// Fetch Procurements API to build supplier metrics
async function loadSuppliersFromAPI() {
    try {
        // Fetch up to 1000 records for aggregation
        const res = await fetch('http://localhost:8000/api/procurements?limit=1000');
        if (!res.ok) throw new Error('API response not ok');
        const json = await res.json();
        const data = json.data || [];
        
        // Group by Supplier_Name
        const supplierMap = {};
        data.forEach(row => {
            if (row.Category !== 'IT Software') return; // Only process IT Software for recommendation module
            
            const name = row.Supplier_Name;
            if (!name) return;
            
            if (!supplierMap[name]) {
                supplierMap[name] = { 
                    name: name, count: 0, sum_savings: 0, 
                    sum_otd: 0, sum_risk: 0, sum_esg: 0,
                    preferred: row.Preferred_Supplier === 'Yes'
                };
            }
            
            supplierMap[name].count++;
            supplierMap[name].sum_savings += parseFloat(row.Savings_Pct || 0);
            supplierMap[name].sum_otd += (row.On_Time_Delivery === 'Yes' ? 100 : 0);
            supplierMap[name].sum_risk += parseFloat(row.Supplier_Risk || 0);
            supplierMap[name].sum_esg += parseFloat(row.Supplier_ESG_Score || 0);
        });
        
        // Calculate averages
        mockSuppliers = Object.values(supplierMap).map(s => {
            return {
                name: s.name,
                savings: +(s.sum_savings / s.count).toFixed(2),
                otd: +(s.sum_otd / s.count).toFixed(1),
                risk: +(s.sum_risk / s.count).toFixed(1),
                esg: +(s.sum_esg / s.count).toFixed(1),
                preferred: s.preferred
            };
        });
        
        // Fallback if no IT Software data
        if (mockSuppliers.length === 0) {
            mockSuppliers = [
                { name: "Demo IT Vendor (No Data)", savings: 5.0, otd: 90, risk: 20, esg: 70, preferred: true }
            ];
        }
        
        // Default render
        if (typeof window.switchScenario === 'function') {
            window.switchScenario('cost');
        }
    } catch (e) {
        console.error("Failed to load procurements API:", e);
    }
}

// Automatically load suppliers on script load
document.addEventListener('DOMContentLoaded', () => {
    loadSuppliersFromAPI();
});

window.switchScenario = function(scenario) {
    // Update button active states
    document.querySelectorAll('.scenario-toggles .btn').forEach(btn => {
        btn.classList.remove('btn-primary');
        btn.classList.add('btn-secondary');
    });
    
    const targetBtn = document.getElementById('btn-scenario-' + scenario);
    if (targetBtn) {
        targetBtn.classList.remove('btn-secondary');
        targetBtn.classList.add('btn-primary');
    }

    renderSuppliers(scenario);
};

function renderSuppliers(scenario) {
    const list = document.getElementById('recommendation-list');
    if (!list) return;

    // Clone array to sort safely
    let sortedSuppliers = [...mockSuppliers];

    sortedSuppliers.sort((a, b) => {
        if (scenario === 'cost') {
            return b.savings - a.savings; // Highest savings first
        } else if (scenario === 'urgent') {
            return b.otd - a.otd; // Highest OTD first
        } else if (scenario === 'compliance') {
            return b.esg - a.esg; // Highest ESG first
        }
        return 0;
    });

    // Take top 3
    sortedSuppliers = sortedSuppliers.slice(0, 3);

    list.innerHTML = '';
    sortedSuppliers.forEach((sup, index) => {
        let score = 0;
        let highlight = '';
        if (scenario === 'cost') {
            score = Math.round(sup.savings * 10); // arbitrary score mapping
            highlight = `節省率優異 (${sup.savings}%)`;
        } else if (scenario === 'urgent') {
            score = sup.otd;
            highlight = `準交率極高 (${sup.otd}%)`;
        } else {
            score = sup.esg;
            highlight = `ESG 領先 (${sup.esg} 分)`;
        }

        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><span class="badge ${index === 0 ? 'tier-1' : ''}">${index + 1}</span></td>
            <td><strong>${sup.name}</strong> ${sup.preferred ? '<i class="ph ph-star" style="color: gold;" title="Preferred"></i>' : ''}</td>
            <td class="success-text">${score} / 100</td>
            <td><span class="alert-tag success" style="padding: 0.2rem 0.5rem;"><i class="ph ph-check-circle"></i> ${highlight}</span></td>
            <td><button class="btn btn-sm btn-primary">選擇</button></td>
        `;
        list.appendChild(tr);
    });
}

// Initial render for Recommendation Model
document.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('view-recommendation')) {
        renderSuppliers('cost'); // Default scenario
    }
});
