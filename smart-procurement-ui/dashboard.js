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
