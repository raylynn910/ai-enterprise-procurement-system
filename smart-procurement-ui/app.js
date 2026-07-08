// app.js

function updateLog(message) {
    const logContent = document.getElementById('log-content');
    const timestamp = new Date().toLocaleTimeString('zh-TW', { hour12: false });
    const logEntry = document.createElement('div');
    logEntry.innerHTML = `<span style="color: #64748b;">[${timestamp}]</span> ${message}`;
    logContent.appendChild(logEntry);
    
    // Auto scroll to bottom
    logContent.scrollTop = logContent.scrollHeight;
}

function logAction(actionType) {
    if (actionType === 'approve_early') {
        updateLog(`[Action] 使用者點擊「確認提早下單」。觸發規則：Risk_Level = High, User_Override = False.`);
        updateLog(`[System] 已送出提早 5 天交貨請求至 ERP 系統。`);
        alert("已送出提早下單請求！");
    } else if (actionType === 'ignore_risk') {
        const reason = prompt("請輸入忽略此高風險警告的理由 (必填):");
        if (reason) {
            updateLog(`[Action] 使用者點擊「忽略風險」。理由: "${reason}"`);
            updateLog(`[System] 警告已解除，訂單進入一般審核流程，並標記需覆核。`);
        } else {
            updateLog(`[Error] 忽略風險失敗：未提供理由。 (防呆機制觸發)`);
            alert("防呆機制：忽略高風險警告必須填寫理由！");
        }
    }
}

// Simulate fetching external PMI data
setTimeout(() => {
    updateLog(`[API] 嘗試從 index.ndc.gov.tw 獲取景氣燈號與 PMI 資料...`);
    
    setTimeout(() => {
        updateLog(`[API] 獲取成功！目前狀態：原物料價格看漲 (紅燈)，建議鎖定價格。`);
        // We could dynamically change the traffic light here if we wanted
    }, 1500);
}, 2000);
