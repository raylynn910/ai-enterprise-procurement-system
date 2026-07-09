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
    
    if (btnPredict) {
        btnPredict.addEventListener('click', async () => {
            // Add simple loading state
            btnPredict.innerHTML = '<i class="ph ph-spinner-gap ph-spin"></i> 預測中...';
            btnPredict.disabled = true;

            const supplier = document.getElementById('input-supplier').value;
            const category = document.getElementById('input-contract') ? document.getElementById('input-contract').value : 'IT Software';
            const quantity = parseInt(document.getElementById('input-quantity').value) || 100;
            const budget = parseFloat(document.getElementById('input-budget').value) || 0;
            const budgetUnitPrice = budget / quantity; // API expects unit price
            
            try {
                // Call POST /api/predict/savings API
                const response = await fetch('http://localhost:8000/api/predict/savings', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        supplier_id: supplier || 'SUP-001',
                        category: category,
                        quantity: quantity,
                        budget_price: budgetUnitPrice
                    })
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const result = await response.json();
                
                let savingsPct = result.pred_savings_pct;
                let savingsAmount = (budget * savingsPct) / 100;
                
                // Update UI
                if (savingsPct > 0) {
                    valPredSavings.innerText = '+' + savingsPct.toFixed(1) + '%';
                    valPredSavings.className = 'result-value success-text';
                } else {
                    valPredSavings.innerText = savingsPct.toFixed(1) + '%';
                    valPredSavings.className = 'result-value danger-text';
                }
                
                valPredAmount.innerText = '$' + savingsAmount.toFixed(2);
                
                // Show recommendation text in a small alert below the amount
                let recDiv = document.getElementById('pred-recommendation');
                if (!recDiv) {
                    recDiv = document.createElement('div');
                    recDiv.id = 'pred-recommendation';
                    recDiv.style.marginTop = '10px';
                    recDiv.style.fontSize = '0.9rem';
                    predictionResult.appendChild(recDiv);
                }
                
                const statusColor = result.pred_class_code === 2 ? 'danger-text' : (result.pred_class_code === 1 ? 'warning-text' : 'success-text');
                
                recDiv.innerHTML = `
                    <strong class="${statusColor}">${result.status_text}</strong><br/>
                    <div style="margin-top: 5px; padding: 5px; background: rgba(255,255,255,0.05); border-radius: 4px; font-size: 0.8rem; color: #94a3b8;">
                        <i class="ph ph-magic-wand"></i> <strong>AI 自動補水特徵:</strong><br/>
                        供應商風險: ${result.enriched_features.supplier_risk} | 越權紀錄: ${result.enriched_features.maverick_spend} | 首選供應商: ${result.enriched_features.preferred_supplier}
                    </div>
                `;
                
                // Show result panel
                predictionResult.style.display = 'block';
                
            } catch (error) {
                console.error('API Error:', error);
                alert('預測 API 呼叫失敗，請確認 FastAPI 後端已啟動且模型載入正常。');
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

// Automatically load suppliers and options on script load
document.addEventListener('DOMContentLoaded', () => {
    loadFormOptionsFromAPI();
    loadSuppliersFromAPI();
});

// Fetch Form Options from Backend
async function loadFormOptionsFromAPI() {
    try {
        const res = await fetch('http://localhost:8000/api/form-options');
        if (!res.ok) throw new Error('Form options API failed');
        const json = await res.json();
        
        // Populate Categories (Contracts)
        const catSelect = document.getElementById('input-contract');
        if (catSelect && json.categories) {
            catSelect.innerHTML = '';
            json.categories.forEach(cat => {
                const opt = document.createElement('option');
                opt.value = cat;
                opt.textContent = cat;
                catSelect.appendChild(opt);
            });
        }
        
        // Populate Suppliers
        const supSelect = document.getElementById('input-supplier');
        if (supSelect && json.suppliers) {
            supSelect.innerHTML = '';
            json.suppliers.forEach(sup => {
                const opt = document.createElement('option');
                opt.value = sup.id; // Send Supplier_ID to the API, not Name!
                opt.textContent = sup.name;
                supSelect.appendChild(opt);
            });
        }
    } catch (e) {
        console.error("Failed to load form options:", e);
    }
}

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

// Fetch Monthly Trend Data and Render Chart
async function loadTrendChart() {
    const ctx = document.getElementById('monthlyTrendChart');
    if (!ctx) return;
    
    try {
        const res = await fetch('http://localhost:8000/api/trends/monthly');
        if (!res.ok) throw new Error('API response not ok');
        const json = await res.json();
        
        const data = json.data || [];
        
        // Format X-axis labels (e.g., "2023 Jan")
        const labels = data.map(row => {
            const monthStr = row.PO_Month ? row.PO_Month.substring(0, 3) : '';
            return `${row.PO_Year} ${monthStr}`;
        });
        
        const savingsData = data.map(row => row.avg_savings_pct);
        const otdData = data.map(row => row.on_time_delivery_rate);
        
        // Cyberpunk Theme Dual-axis Mixed Chart
        new Chart(ctx, {
            data: {
                labels: labels.length > 0 ? labels : ['2023 Jan', '2023 Feb', '2023 Mar'],
                datasets: [
                    {
                        type: 'line',
                        label: '平均節省率 (Average Savings %)',
                        data: savingsData.length > 0 ? savingsData : [5.4, 6.1, 7.2],
                        borderColor: '#10b981', // Green for savings
                        backgroundColor: '#10b981',
                        borderWidth: 3,
                        pointBackgroundColor: '#10b981',
                        pointBorderColor: '#fff',
                        pointHoverBackgroundColor: '#fff',
                        pointHoverBorderColor: '#10b981',
                        pointRadius: 4,
                        tension: 0.4,
                        yAxisID: 'y'
                    },
                    {
                        type: 'bar',
                        label: '供應商準交率 (On-Time Delivery Rate %)',
                        data: otdData.length > 0 ? otdData : [92.5, 89.0, 95.1],
                        backgroundColor: 'rgba(139, 92, 246, 0.6)', // Purple for OTD
                        borderColor: '#8b5cf6',
                        borderWidth: 1,
                        borderRadius: 4,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    legend: {
                        labels: { color: '#94a3b8' }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#94a3b8' }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: { display: true, text: '節省率 (%)', color: '#10b981' },
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#10b981' },
                        min: 0
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: { display: true, text: '準交率 (%)', color: '#8b5cf6' },
                        grid: { drawOnChartArea: false }, // only draw grid lines for one axis to keep it clean
                        ticks: { color: '#8b5cf6' },
                        min: 0,
                        max: 100 // OTD rate max is 100%
                    }
                }
            }
        });
    } catch (e) {
        console.error('Failed to load trend API:', e);
        // Fallback for UI testing if API fails
        new Chart(ctx, {
            data: {
                labels: ['2023 Jan', '2023 Feb', '2023 Mar', '2023 Apr', '2023 May'],
                datasets: [
                    {
                        type: 'line',
                        label: '平均節省率 (Mock API 失敗)',
                        data: [5.2, 5.8, 4.9, 6.5, 7.1],
                        borderColor: '#f43f5e',
                        backgroundColor: '#f43f5e',
                        borderWidth: 2,
                        yAxisID: 'y'
                    },
                    {
                        type: 'bar',
                        label: '供應商準交率 (Mock API 失敗)',
                        data: [85, 88, 92, 90, 94],
                        backgroundColor: 'rgba(244, 63, 94, 0.3)',
                        borderColor: '#f43f5e',
                        borderWidth: 1,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: { 
                responsive: true, 
                maintainAspectRatio: false,
                scales: {
                    y: { position: 'left', min: 0, max: 20 },
                    y1: { position: 'right', min: 0, max: 100, grid: { drawOnChartArea: false } }
                }
            }
        });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    loadTrendChart();
});
