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
        btnPredict.addEventListener('click', () => {
            // Add simple loading state
            btnPredict.innerHTML = '<i class="ph ph-spinner-gap ph-spin"></i> 預測中...';
            btnPredict.disabled = true;

            // Simulate API call delay
            setTimeout(() => {
                // Get form values (can be used to send to backend API later)
                const supplier = document.getElementById('input-supplier').value;
                const budget = parseFloat(document.getElementById('input-budget').value) || 0;
                
                // Mock logic: calculate some fake savings based on inputs
                let fakeSavingsPct = supplier === 'S001' ? 8.5 : (supplier === 'S002' ? -2.1 : 5.0);
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
                
                // Show result panel
                predictionResult.style.display = 'block';
                
                // Reset button
                btnPredict.innerHTML = '<i class="ph ph-magic-wand"></i> 執行預測';
                btnPredict.disabled = false;
            }, 1000);
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

// Supplier Recommendation Model Logic
const mockSuppliers = [
    { name: "CloudTech Solutions", savings: 12.5, otd: 98, risk: 10, esg: 85, preferred: true },
    { name: "Legacy IT Partners", savings: 2.1, otd: 82, risk: 65, esg: 45, preferred: false },
    { name: "Agile Softworks", savings: 8.0, otd: 99, risk: 5, esg: 92, preferred: true },
    { name: "Global Systems Inc.", savings: -1.5, otd: 70, risk: 80, esg: 60, preferred: false },
    { name: "Nordic Office Solutions", savings: 5.5, otd: 95, risk: 20, esg: 89, preferred: true }
];

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
