import re

# 1. Update HTML
with open('smart-procurement-ui/dashboard.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Add button
topbar_target = '<div class="header-actions">'
topbar_replacement = '''<div class="header-actions">
                    <button id="btn-open-add-supplier" class="btn-primary" style="padding: 0.5rem 1rem; border-radius: 8px; font-weight: 600; display: flex; align-items: center; gap: 0.5rem; border: none; cursor: pointer; background: var(--primary); color: #000;">
                        <i class="ph ph-plus-circle"></i> Add Vendor
                    </button>'''
html = html.replace(topbar_target, topbar_replacement)

# Add modal
modal_html = '''
    <!-- Add Supplier Modal -->
    <div id="add-supplier-modal" class="modal-overlay" style="display: none; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(0,0,0,0.6); z-index: 1000; justify-content: center; align-items: center; backdrop-filter: blur(5px);">
        <div class="modal-content glass-panel" style="width: 450px; padding: 2rem; border-radius: 16px; animation: slideIn 0.3s ease;">
            <div class="modal-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
                <h2 style="font-size: 1.5rem; display: flex; align-items: center; gap: 0.5rem; color: #fff;"><i class="ph ph-buildings" style="color: var(--primary);"></i> Onboard New Vendor</h2>
                <button id="btn-close-modal" class="icon-btn" style="background: none; border: none; color: #fff; font-size: 1.5rem; cursor: pointer;"><i class="ph ph-x"></i></button>
            </div>
            <div class="modal-body">
                <form id="add-supplier-form" style="display: flex; flex-direction: column; gap: 1rem;">
                    <div class="form-group" style="display: flex; flex-direction: column; gap: 0.5rem;">
                        <label style="color: var(--text-muted); font-size: 0.9rem;">Supplier Name</label>
                        <input type="text" id="as-name" required placeholder="e.g. Acme Corp" style="padding: 0.75rem; border-radius: 8px; border: 1px solid var(--border-glass); background: rgba(0,0,0,0.3); color: #fff;">
                    </div>
                    <div class="form-group" style="display: flex; flex-direction: column; gap: 0.5rem;">
                        <label style="color: var(--text-muted); font-size: 0.9rem;">Country</label>
                        <input type="text" id="as-country" required placeholder="e.g. Taiwan" style="padding: 0.75rem; border-radius: 8px; border: 1px solid var(--border-glass); background: rgba(0,0,0,0.3); color: #fff;">
                    </div>
                    <div class="form-group" style="display: flex; flex-direction: column; gap: 0.5rem;">
                        <label style="color: var(--text-muted); font-size: 0.9rem;">Category</label>
                        <select id="as-category" required style="padding: 0.75rem; border-radius: 8px; border: 1px solid var(--border-glass); background: #1a1a2e; color: #fff;">
                            <option value="IT Software">IT Software</option>
                            <option value="Hardware">Hardware</option>
                            <option value="Logistics">Logistics</option>
                            <option value="Raw Materials">Raw Materials</option>
                        </select>
                    </div>
                    <div class="form-group" style="display: flex; flex-direction: column; gap: 0.5rem;">
                        <label style="color: var(--text-muted); font-size: 0.9rem;">Initial Risk Level</label>
                        <select id="as-risk" required style="padding: 0.75rem; border-radius: 8px; border: 1px solid var(--border-glass); background: #1a1a2e; color: #fff;">
                            <option value="Low">Low</option>
                            <option value="Medium">Medium</option>
                            <option value="High">High</option>
                        </select>
                    </div>
                    <div class="form-group" style="display: flex; flex-direction: column; gap: 0.5rem;">
                        <label style="color: var(--text-muted); font-size: 0.9rem;">ESG Score (0-100)</label>
                        <input type="number" id="as-esg" required min="0" max="100" value="50" style="padding: 0.75rem; border-radius: 8px; border: 1px solid var(--border-glass); background: rgba(0,0,0,0.3); color: #fff;">
                    </div>
                    <div class="form-actions" style="margin-top: 1.5rem; display: flex; justify-content: flex-end; gap: 1rem;">
                        <button type="button" id="btn-cancel-add" style="padding: 0.75rem 1.5rem; border-radius: 8px; border: 1px solid var(--border-glass); background: transparent; color: #fff; cursor: pointer;">Cancel</button>
                        <button type="submit" id="btn-submit-add" style="padding: 0.75rem 1.5rem; border-radius: 8px; border: none; background: var(--primary); color: #000; font-weight: bold; cursor: pointer;">Submit & AI Evaluate</button>
                    </div>
                </form>
            </div>
        </div>
    </div>
</body>'''
html = html.replace('</body>', modal_html)

with open('smart-procurement-ui/dashboard.html', 'w', encoding='utf-8') as f:
    f.write(html)


# 2. Update JS
js_append = '''

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
'''
with open('smart-procurement-ui/dashboard.js', 'a', encoding='utf-8') as f:
    f.write(js_append)

print("HTML and JS updated successfully.")
