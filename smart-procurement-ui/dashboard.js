// dashboard.js

document.addEventListener('DOMContentLoaded', () => {
    loadOverviewKPIs();
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

    // Onboarding Form Logic
    const btnAssessSupplier = document.getElementById('btn-assess-supplier');
    if (btnAssessSupplier) {
        btnAssessSupplier.addEventListener('click', async () => {
            const name = document.getElementById('ob-name').value;
            const region = document.getElementById('ob-region').value;
            
            if (!name || name.trim().length < 2) {
                alert('【警告】名稱過短，請輸入有效的公司名稱！');
                return;
            }
            if (!isNaN(name.replace(/ /g, ''))) {
                alert('【警告】公司名稱不能只有數字，請確認您輸入的是正確的實體名稱！\n\n系統防呆機制：已阻擋本次異常請求。');
                return;
            }

            btnAssessSupplier.innerHTML = '<i class="ph ph-spinner-gap ph-spin"></i> 徵信掃描中...';
            btnAssessSupplier.disabled = true;
            document.getElementById('onboarding-result').style.display = 'none';

            try {
                const supplierId = document.getElementById('ob-id').value || name;
                const region = document.getElementById('ob-region').value;
                const category = document.getElementById('ob-category').value;
                
                const response = await fetch('http://localhost:8000/api/predict/supplier-risk', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        supplier_id: supplierId,
                        country: region,
                        category: category,
                        lead_time_days: 30
                    })
                });
                
                if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                const result = await response.json();
                
                // --- First-layer Ghost Company Protection (防呆) ---
                if (result.recommendation && result.recommendation.includes("找不到該公司相關資訊")) {
                    alert(`【警告】查無「${name}」的公司登記或情報資料！\n\n系統防呆機制已啟動：請確認公司名稱是否輸入正確，或該公司未在當地合法設立行號。`);
                    btnAssessSupplier.innerHTML = '執行 AI 盡職調查';
                    btnAssessSupplier.disabled = false;
                    document.getElementById('onboarding-result').style.display = 'none';
                    return; // Abort UI rendering
                }
                
                let risk = result.risk_level;
                // Risk score from backend is 0-100. We can map it back to ESG roughly or use it for UI.
                let esgScore = 100 - result.risk_score; // Lower risk score = Higher ESG
                esgScore = Math.min(100, Math.max(0, esgScore));
                
                // Decide status based on API risk level
                let decision = "核准 (Approve)";
                let dClass = "success-text";
                let rClass = "var(--success)";
                let rBg = "rgba(16, 185, 129, 0.2)";
                
                if (risk === "High") {
                    decision = "拒絕 (Reject)";
                    dClass = "danger-text";
                    rClass = "var(--danger)";
                    rBg = "rgba(220, 38, 38, 0.2)";
                } else if (risk === "Medium") {
                    decision = "人工覆核 (Review)";
                    dClass = "warning-text";
                    rClass = "var(--warning)";
                    rBg = "rgba(245, 158, 11, 0.2)";
                }

                document.getElementById('ob-res-decision').innerText = decision;
                document.getElementById('ob-res-decision').className = `result-value ${dClass}`;
                
                const riskBadge = document.getElementById('ob-res-risk');
                riskBadge.innerText = `Risk: ${risk}`;
                riskBadge.style.color = rClass;
                riskBadge.style.borderColor = rClass;
                riskBadge.style.background = rBg;
                
                // Toggle ERP Sync Button
                const btnSyncErp = document.getElementById('btn-sync-erp');
                if (btnSyncErp) {
                    if (risk !== 'High') {
                        btnSyncErp.style.display = 'flex';
                        btnSyncErp.disabled = false;
                        btnSyncErp.innerHTML = '<i class="ph ph-cloud-arrow-up" style="font-size: 1.1rem;"></i> 匯入企業採購系統 (ERP)';
                        btnSyncErp.style.background = 'linear-gradient(90deg, #0ea5e9, #2563eb)';
                    } else {
                        btnSyncErp.style.display = 'none';
                    }
                }

                document.getElementById('ob-res-esgval').innerText = `${esgScore.toFixed(0)} / 100`;
                document.getElementById('ob-res-esgbar').style.width = `${esgScore}%`;
                
                // Update news based on recommendation
                if (risk === "High") {
                    document.getElementById('ob-icon-news').className = "ph ph-warning-circle";
                    document.getElementById('ob-icon-news').style.color = "var(--danger)";
                    document.getElementById('ob-log-news').innerText = result.recommendation;
                } else {
                    document.getElementById('ob-icon-news').className = "ph ph-check-circle";
                    document.getElementById('ob-icon-news').style.color = "var(--success)";
                    document.getElementById('ob-log-news').innerText = result.recommendation;
                }
                
                if (result.recommendation && result.recommendation.includes("合規阻斷")) {
                    document.getElementById('ob-icon-sanc').className = "ph ph-warning-circle";
                    document.getElementById('ob-icon-sanc').style.color = "var(--danger)";
                    document.getElementById('ob-log-sanc').innerText = "警告：於全球制裁名單或 PEP 黑名單發現疑似相符紀錄，合規性不予通過。";
                } else {
                    document.getElementById('ob-icon-sanc').className = "ph ph-check-circle";
                    document.getElementById('ob-icon-sanc').style.color = "var(--success)";
                    document.getElementById('ob-log-sanc').innerText = "未比對到全球制裁名單或 PEP 黑名單。";
                }

                document.getElementById('ob-log-corp').innerText = `實體名稱 '${name}' 已於當地商業司註冊，狀態活躍 (Active)。`;

                // --- OSINT Modal Logic ---
                const btnViewOsint = document.getElementById('btn-view-osint');
                if (btnViewOsint) {
                    const osintModal = document.getElementById('osint-modal');
                    const closeOsintModal = document.getElementById('close-osint-modal');
                    const osintContainer = document.getElementById('osint-sources-container');
                    
                    if (result.osint_sources && result.osint_sources.length > 0) {
                        btnViewOsint.style.display = 'inline-block';
                        btnViewOsint.onclick = function() {
                            osintContainer.innerHTML = '';
                            result.osint_sources.forEach(src => {
                                const card = document.createElement('div');
                                card.style.background = 'rgba(255, 255, 255, 0.05)';
                                card.style.padding = '1rem';
                                card.style.borderRadius = '8px';
                                card.style.border = '1px solid rgba(255,255,255,0.05)';
                                card.innerHTML = `
                                    <h4 style="margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 500;">
                                        <a href="${src.url}" target="_blank" style="color: var(--primary); text-decoration: none; display: flex; align-items: center; gap: 4px;">
                                            ${src.title} <i class="ph ph-arrow-square-out" style="font-size: 0.9rem;"></i>
                                        </a>
                                    </h4>
                                    <p style="margin: 0; font-size: 0.85rem; color: var(--text-muted); line-height: 1.5;">${src.snippet}</p>
                                `;
                                osintContainer.appendChild(card);
                            });
                            osintModal.style.display = 'flex';
                        };
                    } else {
                        btnViewOsint.style.display = 'none';
                        btnViewOsint.onclick = null;
                    }
                    
                    if (closeOsintModal && osintModal) {
                        closeOsintModal.onclick = function() {
                            osintModal.style.display = 'none';
                        };
                        osintModal.onclick = function(e) {
                            if (e.target === osintModal) {
                                osintModal.style.display = 'none';
                            }
                        };
                    }
                }

                document.getElementById('onboarding-result').style.display = 'block';
                document.getElementById('onboarding-result').style.borderTopColor = rClass;
                
                // Render ApexCharts Radar Chart
                if (window.obRadarChart) {
                    window.obRadarChart.destroy();
                }
                
                // Read the 5 dimensions calculated by the backend ML API
                const reputationScore = result.reputation_score || 0;
                const financialScore = result.financial_score || 0;
                const deliveryScore = result.delivery_score || 0;
                const esgDimScore = result.esg_score || 0;
                const pricingScore = result.pricing_score || 0;

                const options = {
                    series: [{
                        name: '供應商能力',
                        data: [
                            Math.round(reputationScore), 
                            Math.round(financialScore), 
                            Math.round(deliveryScore), 
                            Math.round(esgDimScore), 
                            Math.round(pricingScore)
                        ]
                    }],
                    chart: {
                        height: 350,
                        type: 'radar',
                        toolbar: { show: false },
                        dropShadow: {
                            enabled: true, blur: 1, left: 1, top: 1
                        },
                        background: 'transparent'
                    },
                    stroke: {
                        width: 2,
                        colors: ['#10b981']
                    },
                    fill: {
                        opacity: 0.4,
                        colors: ['#10b981']
                    },
                    markers: {
                        size: 4,
                        colors: ['#1a202c'],
                        strokeColors: '#10b981',
                        strokeWidth: 2,
                    },
                    xaxis: {
                        categories: ['市場聲譽 (Reputation)', '財務穩定 (Financial)', '交期可靠 (Delivery)', '永續發展 (ESG)', '價格競爭力 (Pricing)'],
                        labels: {
                            style: {
                                colors: ['#9ca3af', '#9ca3af', '#9ca3af', '#9ca3af', '#9ca3af'],
                                fontSize: '11px',
                                fontFamily: 'Inter, sans-serif'
                            }
                        }
                    },
                    yaxis: {
                        min: 0,
                        max: 100,
                        tickAmount: 5,
                        show: false
                    },
                    plotOptions: {
                        radar: {
                            polygons: {
                                strokeColors: 'rgba(255,255,255,0.1)',
                                connectorColors: 'rgba(255,255,255,0.1)',
                                fill: {
                                    colors: ['transparent', 'transparent']
                                }
                            }
                        }
                    },
                    theme: {
                        mode: 'dark'
                    },
                    tooltip: {
                        theme: 'dark'
                    }
                };
                
                // Update colors based on risk
                if(risk === "High") {
                    options.stroke.colors = ['#ef4444'];
                    options.fill.colors = ['#ef4444'];
                    options.markers.strokeColors = '#ef4444';
                } else if(risk === "Medium") {
                    options.stroke.colors = ['#f59e0b'];
                    options.fill.colors = ['#f59e0b'];
                    options.markers.strokeColors = '#f59e0b';
                }

                window.obRadarChart = new ApexCharts(document.querySelector("#ob-radar-chart"), options);
                window.obRadarChart.render();
                
            } catch(error) {
                console.error("Prediction API Error:", error);
                alert("徵信掃描失敗，請確認後端 API 已啟動。");
            } finally {
                btnAssessSupplier.innerHTML = '<i class="ph ph-magnifying-glass"></i> 執行 AI 徵信掃描';
                btnAssessSupplier.disabled = false;
            }
        });
    }

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
    let reportPieChartInstance = null;
    let reportBarChartInstance = null;

    document.getElementById('btn-weekly-report')?.addEventListener('click', async () => {
        const btn = document.getElementById('btn-weekly-report');
        const loadingStatus = document.getElementById('report-loading-status');
        const dashboard = document.getElementById('report-dashboard');
        const placeholder = document.getElementById('report-placeholder');
        
        btn.disabled = true;
        loadingStatus.style.display = 'inline-block';
        
        try {
            const res = await fetch('http://localhost:8000/api/reports/weekly');
            if (!res.ok) throw new Error('API fetching failed');
            const result = await res.json();
            
            // Render Markdown text
            document.getElementById('val-report-content').innerHTML = marked.parse(result.markdown);
            
            // Show dashboard, hide placeholder
            placeholder.style.display = 'none';
            dashboard.style.display = 'grid';
            
            // Chart 1: Pie Chart (Risk Distribution)
            const pieCtx = document.getElementById('report-pie-chart');
            if (reportPieChartInstance) reportPieChartInstance.destroy();
            reportPieChartInstance = new Chart(pieCtx, {
                type: 'doughnut',
                data: {
                    labels: result.charts.pie.labels,
                    datasets: [{
                        data: result.charts.pie.data,
                        backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
                        borderWidth: 0,
                        hoverOffset: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'bottom', labels: { color: '#e2e8f0' } }
                    }
                }
            });

            // Chart 2: Bar Chart (Cost Avoidance Trend)
            const barCtx = document.getElementById('report-bar-chart');
            if (reportBarChartInstance) reportBarChartInstance.destroy();
            reportBarChartInstance = new Chart(barCtx, {
                type: 'bar',
                data: {
                    labels: result.charts.bar.labels,
                    datasets: [{
                        label: 'Cost Avoidance (USD)',
                        data: result.charts.bar.data,
                        backgroundColor: 'rgba(56, 189, 248, 0.6)',
                        borderColor: '#38bdf8',
                        borderWidth: 1,
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: { color: '#94a3b8' },
                            grid: { color: 'rgba(255,255,255,0.05)' }
                        },
                        x: {
                            ticks: { color: '#94a3b8' },
                            grid: { display: false }
                        }
                    },
                    plugins: {
                        legend: { display: false }
                    }
                }
            });

        } catch (e) {
            console.error(e);
            alert('報表生成失敗，請確認後端已連線。');
        } finally {
            btn.disabled = false;
            loadingStatus.style.display = 'none';
        }
    });

    document.getElementById('btn-monthly-report')?.addEventListener('click', () => {
        alert('報表生成中... (此處可串接後端 API 下載 Excel/PDF)');
    });
});

// Global suppliers array populated from API
let mockSuppliers = [];

// Fetch Procurements API to build supplier metrics and overview KPIs
async function loadDashboardData() {
    try {
        // 1. Fetch Risk Orders for High Risk Count
        let highRiskCount = 0;
        try {
            const riskRes = await fetch('http://localhost:8000/api/risk/orders');
            if (riskRes.ok) {
                const riskData = await riskRes.json();
                if (riskData.data) highRiskCount = riskData.data.length;
            }
        } catch(e) { console.warn("Failed to fetch risk orders:", e); }

        // 2. Fetch up to 2000 records for aggregation
        const res = await fetch('http://localhost:8000/api/procurements?limit=2000');
        if (!res.ok) throw new Error('API response not ok');
        const json = await res.json();
        const data = json.data || [];
        
        let sumSavingsPct = 0;
        let sumEsg = 0;
        let sumRiskNum = 0;
        let maverickCount = 0;
        let validRecords = 0;

        // Group by Supplier_Name
        const supplierMap = {};
        
        data.forEach(row => {
            const name = row.Supplier_Name;
            if (!name) return;
            
            // Overall KPIs
            const savings = parseFloat(row.Savings_Pct) || 0;
            const esg = parseFloat(row.Supplier_ESG_Score) || 0;
            const isMaverick = row.Maverick_Spend === 'Yes';
            
            let riskNum = 50; // default medium
            if (row.Supplier_Risk === 'Low') riskNum = 10;
            if (row.Supplier_Risk === 'High') riskNum = 90;

            sumSavingsPct += savings;
            sumEsg += esg;
            sumRiskNum += riskNum;
            if (isMaverick) maverickCount++;
            validRecords++;
            
            if (!supplierMap[name]) {
                supplierMap[name] = { 
                    name: name, count: 0, sum_savings: 0, 
                    sum_otd: 0, sum_risk: 0, sum_esg: 0,
                    preferred: row.Preferred_Supplier === 'Yes',
                    category: row.Category,
                    controversies: 0,
                    risk_level: row.Supplier_Risk || 'Medium'
                };
            }
            
            supplierMap[name].count++;
            supplierMap[name].sum_savings += savings;
            supplierMap[name].sum_otd += (row.On_Time_Delivery === 'Yes' ? 100 : 0);
            supplierMap[name].sum_risk += riskNum;
            supplierMap[name].sum_esg += esg;
            if (isMaverick || row.Days_Late > 10) {
                supplierMap[name].controversies++;
            }
        });
        
        // Populate Overview KPIs
        if (validRecords > 0) {
            const avgSavings = (sumSavingsPct / validRecords).toFixed(1);
            const avgEsg = (sumEsg / validRecords).toFixed(0);
            const avgRisk = (sumRiskNum / validRecords).toFixed(0);
            
            const elSavings = document.getElementById('val-est-savings');
            if (elSavings) {
                const sign = avgSavings > 0 ? '+' : '';
                const trendClass = avgSavings > 0 ? 'trend up' : 'trend down';
                const trendIcon = avgSavings > 0 ? 'ph-trend-up' : 'ph-trend-down';
                elSavings.innerHTML = `${avgSavings}% <span class="${trendClass}"><i class="ph ${trendIcon}"></i></span>`;
                if (avgSavings <= 0) elSavings.style.color = 'var(--danger)';
            }
            
            const elRisk = document.getElementById('val-avg-risk');
            if (elRisk) {
                let status = avgRisk < 30 ? '<span class="status-tag success">安全</span>' : (avgRisk < 70 ? '<span class="status-tag warning">普通</span>' : '<span class="status-tag danger">危險</span>');
                elRisk.innerHTML = `${avgRisk} / 100 ${status}`;
            }

            const elHighRisk = document.getElementById('val-high-risk-count');
            if (elHighRisk) elHighRisk.innerHTML = `${highRiskCount} 筆 <span class="status-tag danger">需關注</span>`;

            const elEsg = document.getElementById('val-avg-esg');
            if (elEsg) elEsg.innerHTML = `${avgEsg} / 100 <span class="status-tag success">健康</span>`;

            const elMaverick = document.getElementById('val-maverick-count');
            if (elMaverick) elMaverick.innerHTML = `${maverickCount} 筆 <span class="status-tag danger">需關注</span>`;
        }
        
        const allSuppliers = Object.values(supplierMap).map(s => {
            return {
                name: s.name,
                category: s.category,
                savings: +(s.sum_savings / s.count).toFixed(2),
                otd: +(s.sum_otd / s.count).toFixed(1),
                riskNum: +(s.sum_risk / s.count).toFixed(1),
                riskLevel: s.risk_level,
                esg: +(s.sum_esg / s.count).toFixed(1),
                preferred: s.preferred,
                controversies: s.controversies
            };
        });

        // 3. Populate Supplier Analysis Table
        const supplierTableBody = document.getElementById('supplier-table-body');
        if (supplierTableBody) {
            supplierTableBody.innerHTML = '';
            // Sort by risk descending
            const tableSuppliers = [...allSuppliers].sort((a,b) => b.riskNum - a.riskNum).slice(0, 15);
            tableSuppliers.forEach(sup => {
                let riskBadge = '';
                if (sup.riskLevel === 'Low') riskBadge = '<span class="badge success">Low</span>';
                else if (sup.riskLevel === 'Medium') riskBadge = '<span class="badge warning">Medium</span>';
                else riskBadge = '<span class="badge danger">High</span>';

                let controText = sup.controversies > 0 ? `${sup.controversies} 筆紀錄` : '無';

                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><i class="ph ph-buildings"></i> ${sup.name}</td>
                    <td>${riskBadge}</td>
                    <td>${sup.esg.toFixed(0)} / 100</td>
                    <td>${sup.otd}%</td>
                    <td class="${sup.controversies > 0 ? 'danger-text' : ''}">${controText}</td>
                `;
                supplierTableBody.appendChild(tr);
            });
        }

        // 4. Update mockSuppliers for the Recommendation module (IT Software only)
        mockSuppliers = allSuppliers.filter(s => s.category === 'IT Software');
        
        // Fallback if no IT Software data
        if (mockSuppliers.length === 0) {
            mockSuppliers = [
                { name: "Demo IT Vendor (No Data)", savings: 5.0, otd: 90, riskNum: 20, esg: 70, preferred: true }
            ];
        }

        // Default render for recommendation
        if (typeof window.switchScenario === 'function') {
            window.switchScenario('cost');
        }
    } catch (e) {
        console.error("Failed to load dashboard data:", e);
    }
}

// Automatically load suppliers and options on script load
document.addEventListener('DOMContentLoaded', () => {
    loadFormOptionsFromAPI();
    loadDashboardData();
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

window.currentScenario = 'cost';

window.switchScenario = async function(scenario) {
    window.currentScenario = scenario;
    
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

    await fetchAndRenderSuppliers();
};

let recommendationRadarChart = null;
let recommendationBarChart = null;

async function fetchAndRenderSuppliers() {
    const podiumList = document.getElementById('recommendation-list');
    const loadingState = document.getElementById('recommendation-loading');
    if (!podiumList) return;
    
    const categorySelect = document.getElementById('recommendation-category');
    const category = categorySelect ? categorySelect.value : 'IT Software';
    const scenario = window.currentScenario || 'cost';

    podiumList.style.display = 'none';
    loadingState.style.display = 'block';

    try {
        const response = await fetch(`http://localhost:8000/api/recommend/suppliers?category=${encodeURIComponent(category)}&scenario=${encodeURIComponent(scenario)}`);
        if (!response.ok) throw new Error('API fetch failed');
        const data = await response.json();

        loadingState.style.display = 'none';
        podiumList.style.display = 'flex';
        podiumList.innerHTML = '';
        
        if (data.length === 0) {
            podiumList.innerHTML = '<div style="text-align: center; color: #a0aec0;">該品項目前沒有符合條件的供應商</div>';
            return;
        }

        // Render Podium Cards
        data.forEach((sup, index) => {
            const isTop1 = index === 0;
            const bgGradient = isTop1 ? 'linear-gradient(135deg, rgba(255,215,0,0.1) 0%, rgba(255,215,0,0.02) 100%)' : 'rgba(255, 255, 255, 0.05)';
            const borderStyle = isTop1 ? 'border: 1px solid rgba(255, 215, 0, 0.3);' : 'border: 1px solid rgba(255, 255, 255, 0.1);';
            const shadowStyle = isTop1 ? 'box-shadow: 0 4px 20px rgba(255, 215, 0, 0.15);' : '';
            
            const card = document.createElement('div');
            card.style = `background: ${bgGradient}; ${borderStyle} ${shadowStyle} border-radius: 12px; padding: 1.5rem; display: flex; align-items: center; justify-content: space-between;`;
            
            card.innerHTML = `
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <div style="width: 50px; height: 50px; border-radius: 50%; background: ${isTop1 ? '#ffd700' : '#4a5568'}; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; font-weight: bold; color: ${isTop1 ? '#000' : '#fff'};">
                        ${isTop1 ? '<i class="ph ph-crown"></i>' : `#${sup.rank}`}
                    </div>
                    <div>
                        <h3 style="margin: 0; font-size: ${isTop1 ? '1.5rem' : '1.25rem'}; color: ${isTop1 ? '#ffd700' : '#fff'};">${sup.name}</h3>
                        <p style="margin: 0.25rem 0 0 0; color: #a0aec0; font-size: 0.9rem;">${sup.country}</p>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 1.25rem; font-weight: bold; color: #4ade80;">${sup.score_text}</div>
                    <div style="font-size: 0.85rem; color: #a0aec0; margin-top: 0.25rem;"><i class="ph ph-info"></i> ${sup.reason}</div>
                    <button class="btn btn-primary btn-sm" style="margin-top: 0.5rem; padding: 0.25rem 1rem;">選擇</button>
                </div>
            `;
            podiumList.appendChild(card);
        });

        // Prepare Chart Data
        const supplierNames = data.map(d => d.name);
        const savingsData = data.map(d => d.raw_metrics.savings_pct);
        const otdData = data.map(d => 100 - d.raw_metrics.days_late * 5); // Rough normalization for OTD score (100 - late*5)
        const esgData = data.map(d => d.raw_metrics.esg_score);
        
        // Create Intuitive "RPG-Style" Radar Chart
        // We swap axes to be Dimensions (Savings, OTD, ESG), and series to be Suppliers.
        const radarSeries = data.map((d, index) => {
            // Normalize all scores to a 0-100 scale for balanced radar display
            const savingsScore = Math.min(100, Math.max(0, d.raw_metrics.savings_pct * 8)); // 12.5% savings = 100 score
            const otdScore = Math.max(0, 100 - d.raw_metrics.days_late * 5);
            const esgScore = d.raw_metrics.esg_score;
            return {
                name: d.name,
                data: [savingsScore.toFixed(1), otdScore.toFixed(1), esgScore.toFixed(1)]
            };
        });

        const customColors = ['#00f0ff', '#b026ff', '#ffb300']; // Cyberpunk primary, purple, warning

        const radarOptions = {
            series: radarSeries,
            chart: { 
                type: 'radar', 
                height: 350, 
                toolbar: { show: false }, 
                background: 'transparent',
                dropShadow: {
                    enabled: true,
                    blur: 8,
                    left: 0,
                    top: 0,
                    opacity: 0.5
                }
            },
            colors: customColors,
            labels: ['Cost Savings Score', 'On-Time Delivery', 'ESG Sustainability'],
            stroke: { width: 3, curve: 'smooth' },
            fill: { opacity: 0.25 },
            markers: { size: 5, hover: { size: 8 } },
            xaxis: {
                labels: {
                    style: { colors: ['#00f0ff', '#39ff14', '#ffb300'], fontSize: '12px', fontFamily: 'Inter', fontWeight: 600 }
                }
            },
            yaxis: { show: false, min: 0, max: 100 },
            theme: { mode: 'dark' },
            legend: { 
                position: 'bottom', 
                labels: { colors: '#fff' },
                markers: { radius: 12 }
            }
        };

        if (recommendationRadarChart) {
            recommendationRadarChart.destroy();
        }
        recommendationRadarChart = new ApexCharts(document.querySelector("#radar-chart"), radarOptions);
        recommendationRadarChart.render();

        // Render or Update Bar Chart
        let primaryMetricData = [];
        let primaryMetricLabel = '';
        let primaryColor = '#4ade80';

        if (scenario === 'cost') {
            primaryMetricData = savingsData;
            primaryMetricLabel = 'Avg Savings (%)';
        } else if (scenario === 'urgent') {
            primaryMetricData = data.map(d => d.raw_metrics.days_late);
            primaryMetricLabel = 'Avg Days Late';
            primaryColor = '#facc15';
        } else {
            primaryMetricData = esgData;
            primaryMetricLabel = 'ESG Score';
            primaryColor = '#60a5fa';
        }

        const barOptions = {
            series: [{ name: primaryMetricLabel, data: primaryMetricData }],
            chart: { type: 'bar', height: 250, toolbar: { show: false }, background: 'transparent' },
            plotOptions: { bar: { horizontal: true, borderRadius: 4 } },
            dataLabels: { enabled: true },
            xaxis: { categories: supplierNames, labels: { style: { colors: '#a0aec0' } } },
            yaxis: { labels: { style: { colors: '#fff', fontSize: '12px' } } },
            colors: [primaryColor],
            theme: { mode: 'dark' }
        };

        if (recommendationBarChart) {
            recommendationBarChart.destroy();
        }
        recommendationBarChart = new ApexCharts(document.querySelector("#bar-chart"), barOptions);
        recommendationBarChart.render();

    } catch (e) {
        console.error("Failed to fetch recommendations:", e);
        loadingState.innerHTML = '<div style="color: #ef4444;"><i class="ph ph-warning-circle" style="font-size: 2rem;"></i><p>載入失敗，請確認後端 API 是否正常運作。</p></div>';
    }
}

// Initial render for Recommendation Model
document.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('view-recommendation')) {
        switchScenario('cost'); // Default scenario
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
    fetchRiskOrders();
});

// --- Team 1 Risk Model Integration ---
async function fetchRiskOrders() {
    try {
        const response = await fetch('http://localhost:8000/api/risk/orders');
        const result = await response.json();
        
        if (result.status === 'success') {
            const tbody = document.getElementById('risk-table-body');
            tbody.innerHTML = ''; // 清空
            
            if (result.data.length === 0) {
                tbody.innerHTML = '<tr><td colspan="6" class="text-center">目前無高風險訂單。</td></tr>';
                return;
            }
            
            result.data.forEach(order => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><strong>${order.po_number}</strong></td>
                    <td>${order.supplier_name}</td>
                    <td class="danger-text">${order.savings_pct}</td>
                    <td><span class="status-tag ${order.maverick === 'Yes' ? 'danger' : 'success'}">${order.maverick}</span></td>
                    <td><span class="status-tag ${order.single_source === 'Yes' ? 'danger' : 'success'}">${order.single_source}</span></td>
                    <td>
                        <button class="btn btn-sm btn-secondary" onclick="alert('【系統提示】\\n模型判定為高風險。建議審查合約。')"><i class="ph ph-handshake"></i> 審查合約</button>
                    </td>
                `;
                tbody.appendChild(tr);
            });
        }
    } catch (e) {
        console.error('Failed to fetch risk orders:', e);
    }
}


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


// Setup ERP Sync Button Event
document.addEventListener('DOMContentLoaded', () => {
    // We can just add event delegation to document since the button might be dynamically hidden/shown
    document.addEventListener('click', (e) => {
        const btnSyncErp = e.target.closest('#btn-sync-erp');
        if (btnSyncErp && !btnSyncErp.disabled) {
            btnSyncErp.disabled = true;
            btnSyncErp.innerHTML = '<i class="ph ph-spinner-gap ph-spin" style="font-size: 1.1rem;"></i> 同步至 ERP 中...';
            btnSyncErp.style.background = 'rgba(255,255,255,0.1)';
            
            setTimeout(() => {
                const fakeVendorId = 'V-' + Math.floor(10000 + Math.random() * 90000);
                btnSyncErp.innerHTML = `<i class="ph ph-check-circle" style="font-size: 1.1rem;"></i> 已成功建檔 (Vendor ID: ${fakeVendorId})`;
                btnSyncErp.style.background = 'var(--success)';
            }, 2000);
        }
    });
});


// Global Search Logic
document.addEventListener('DOMContentLoaded', () => {
    const searchInput = document.getElementById('global-search-input');
    if (searchInput) {
        searchInput.addEventListener('keypress', async (e) => {
            if (e.key === 'Enter') {
                const query = searchInput.value.trim();
                if (!query) return;
                
                try {
                    const res = await fetch(`http://localhost:8000/api/supplier/search?q=${encodeURIComponent(query)}`);
                    const data = await res.json();
                    
                    if (!data.found) {
                        alert(data.message || '找不到符合條件的供應商');
                        return;
                    }
                    
                    // Switch to view-supplier by triggering the nav item click
                    const supplierNav = document.querySelector('.nav-item[data-target="view-supplier"]');
                    if (supplierNav) {
                        supplierNav.click();
                    }
                    // Update Scorecard Data
                    document.getElementById('sc-supplier-name').innerText = data.supplier.name;
                    
                    // Risk Badge
                    const rBadge = document.getElementById('sc-supplier-risk');
                    rBadge.innerText = `Risk: ${data.supplier.risk_level}`;
                    if (data.supplier.risk_level === 'Low') {
                        rBadge.style = 'font-size:1.1rem; padding:0.5rem 1rem; background: rgba(16, 185, 129, 0.2); color: var(--success); border: 1px solid var(--success);';
                    } else if (data.supplier.risk_level === 'High') {
                        rBadge.style = 'font-size:1.1rem; padding:0.5rem 1rem; background: rgba(239, 68, 68, 0.2); color: var(--danger); border: 1px solid var(--danger);';
                    } else {
                        rBadge.style = 'font-size:1.1rem; padding:0.5rem 1rem; background: rgba(245, 158, 11, 0.2); color: var(--warning); border: 1px solid var(--warning);';
                    }
                    
                    // KPI Cards
                    document.getElementById('sc-total-spend').innerText = '$' + Number(data.metrics.total_spend).toLocaleString();
                    document.getElementById('sc-total-pos').innerText = data.metrics.total_pos;
                    
                    const avgSav = document.getElementById('sc-avg-savings');
                    avgSav.innerText = data.metrics.avg_savings + '%';
                    avgSav.className = data.metrics.avg_savings > 0 ? 'kpi-value success-text' : 'kpi-value danger-text';
                    
                    const avgLate = document.getElementById('sc-avg-late');
                    avgLate.innerText = data.metrics.avg_days_late + ' 天';
                    avgLate.className = data.metrics.avg_days_late > 0 ? 'kpi-value warning-text' : 'kpi-value success-text';
                    
                    const esg = document.getElementById('sc-esg-score');
                    esg.innerText = data.supplier.esg_score;
                    esg.className = data.supplier.esg_score >= 70 ? 'kpi-value success-text' : 'kpi-value danger-text';
                    
                    // Recent POs
                    const tbody = document.getElementById('sc-po-tbody');
                    tbody.innerHTML = '';
                    
                    if (data.recent_pos && data.recent_pos.length > 0) {
                        data.recent_pos.forEach(po => {
                            let mavClass = (po.Maverick_Spend.toLowerCase() === 'yes' || po.Maverick_Spend === 'true' || po.Maverick_Spend == 1) ? 'danger-text' : 'success-text';
                            let mavText = (po.Maverick_Spend.toLowerCase() === 'yes' || po.Maverick_Spend === 'true' || po.Maverick_Spend == 1) ? 'Yes' : 'No';
                            
                            const tr = document.createElement('tr');
                            tr.innerHTML = `
                                <td>${po.PO_ID}</td>
                                <td>${po.PO_Date}</td>
                                <td>${po.Category}</td>
                                <td>$${Number(po.Spend).toLocaleString()}</td>
                                <td class="${mavClass}">${mavText}</td>
                                <td>${po.PO_Status}</td>
                            `;
                            tbody.appendChild(tr);
                        });
                    } else {
                        tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; padding: 2rem;">此供應商尚無歷史採購紀錄。</td></tr>';
                    }
                    
                } catch (e) {
                    console.error('Search failed:', e);
                    alert('搜尋時發生錯誤，請稍後再試。');
                }
            }
        });
    }
});


// Add Supplier Modal Logic
document.addEventListener('DOMContentLoaded', () => {
    const modal = document.getElementById('add-supplier-modal');
    const btnOpen = document.getElementById('btn-open-add-supplier');
    const btnClose = document.getElementById('btn-close-modal');
    const btnCancel = document.getElementById('btn-cancel-add');
    const form = document.getElementById('add-supplier-form');
    const btnSubmit = document.getElementById('btn-submit-add');

    if (btnOpen) {
        btnOpen.addEventListener('click', () => {
            modal.style.display = 'flex';
        });
    }

    const closeModal = () => {
        modal.style.display = 'none';
        form.reset();
    };

    if (btnClose) btnClose.addEventListener('click', closeModal);
    if (btnCancel) btnCancel.addEventListener('click', closeModal);

    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            btnSubmit.innerHTML = 'Processing...';
            btnSubmit.disabled = true;

            const payload = {
                name: document.getElementById('as-name').value,
                country: document.getElementById('as-country').value,
                category: document.getElementById('as-category').value,
                risk_level: document.getElementById('as-risk').value,
                esg_score: parseFloat(document.getElementById('as-esg').value)
            };

            try {
                const res = await fetch('http://localhost:8000/api/supplier/add', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (data.success) {
                    alert('Vendor onboarded successfully! AI initial evaluation complete. You can now search for this vendor.');
                    closeModal();
                    // Optionally refresh the current view if needed
                } else {
                    alert('Error: ' + data.message);
                }
            } catch (err) {
                console.error(err);
                alert('API Request Failed');
            } finally {
                btnSubmit.innerHTML = 'Submit & AI Evaluate';
                btnSubmit.disabled = false;
            }
        });
    }
});
