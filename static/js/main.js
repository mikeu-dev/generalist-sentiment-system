// Toast Notification System
function showToast(message, type = 'info') {
    let container = document.querySelector('.toast-container');
    if (!container) {
        container = document.createElement('div');
        container.className = 'toast-container';
        document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    // Add show class after a small delay for transition
    setTimeout(() => toast.classList.add('show'), 10);

    toast.innerHTML = `
        <span class="toast-message">${message}</span>
        <button class="toast-close">&times;</button>
    `;

    // Close button
    toast.querySelector('.toast-close').addEventListener('click', () => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    });

    // Auto close
    setTimeout(() => {
        if (toast.parentElement) {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }
    }, 5000);

    container.appendChild(toast);
}

// Model Selection Logic
function selectModel(element, value) {
    // UI Update
    const group = element.closest('.radio-group');
    if (!group) return;

    group.querySelectorAll('.radio-card').forEach(card => card.classList.remove('selected'));
    element.classList.add('selected');

    // Check hidden radio
    const radio = element.querySelector('input[type="radio"]');
    if (radio) radio.checked = true;
}

document.addEventListener('DOMContentLoaded', () => {
    // File Input Helper
    const fileInputs = document.querySelectorAll('.file-input');
    fileInputs.forEach(input => {
        input.addEventListener('change', (e) => {
            const fileName = e.target.files[0]?.name || "Pilih File";
            e.target.parentNode.querySelector('.file-msg').textContent = fileName;
        });
    });

    // Analyze Form Handler
    document.getElementById('analyze-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const form = e.target;
        const formData = new FormData(form);
        const loading = document.getElementById('analyze-loading');
        const resultsContainer = document.getElementById('results-container');

        loading.classList.remove('hidden');
        resultsContainer.classList.add('hidden');

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                renderResults(data);
                resultsContainer.classList.remove('hidden');
                showToast(`Analisis berhasil! ${data.total} data diproses.`, 'success');
            } else {
                showToast('Error: ' + (data.error || 'Unknown error'), 'error');
            }
        } catch (err) {
            showToast('Request failed: ' + err.message, 'error');
        } finally {
            loading.classList.add('hidden');
        }
    });

    // Search Form Handler
    document.getElementById('search-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const form = e.target;
        const query = form.querySelector('input[name="query"]').value;
        const loading = document.getElementById('search-loading');
        const resultsContainer = document.getElementById('results-container');
        const logContainer = document.getElementById('process-log');

        loading.classList.remove('hidden');
        resultsContainer.classList.add('hidden');

        // Reset Log
        logContainer.innerHTML = '';
        const addLog = (msg, type = 'normal') => {
            const line = document.createElement('div');
            line.className = `log-line ${type}`;
            line.textContent = `> ${msg}`;
            logContainer.appendChild(line);
            logContainer.scrollTop = logContainer.scrollHeight;
        };

        addLog(`System Initialized.`, 'info');
        await new Promise(r => setTimeout(r, 500));
        addLog(`Connecting to Search Engine...`, 'info');
        await new Promise(r => setTimeout(r, 800));
        addLog(`Searching dataset for: "${query}"...`, 'warning');

        // Simulate "Searching" ticker
        let searchInterval = setInterval(() => {
            addLog(`... retrieving data packets ...`);
        }, 2000);

        try {
            const startTime = Date.now();
            const response = await fetch('/search_and_analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: query,
                    limit: form.querySelector('select[name="limit"]').value,
                    model_type: document.querySelector('input[name="model_type"]:checked')?.value || 'default'
                })
            });

            clearInterval(searchInterval);
            const data = await response.json();

            // Ensure visual delay if response was too fast
            const elapsed = Date.now() - startTime;
            if (elapsed < 2000) await new Promise(r => setTimeout(r, 2000 - elapsed));

            if (response.ok) {
                addLog(`Data acquired: ${data.total} records found.`, 'success');
                await new Promise(r => setTimeout(r, 600));

                // Visualize Preprocessing Steps
                addLog(`INITIATING TEXT PREPROCESSOR...`, 'process');
                await new Promise(r => setTimeout(r, 500));

                addLog(`[1/5] Case Folding (Lowercasing)... OK`, 'success');
                await new Promise(r => setTimeout(r, 300));

                addLog(`[2/5] Cleaning Special Characters... OK`, 'success');
                await new Promise(r => setTimeout(r, 300));

                addLog(`[3/5] Normalizing Slang Words (Kamus Alay)... OK`, 'success');
                await new Promise(r => setTimeout(r, 400));

                addLog(`[4/5] Removing Stopwords (Sastrawi)... OK`, 'success');
                await new Promise(r => setTimeout(r, 400));

                addLog(`[5/5] Stemming & Tokenizing... OK`, 'success');
                await new Promise(r => setTimeout(r, 500));

                addLog(`PREPROCESSING COMPLETE.`, 'process');
                await new Promise(r => setTimeout(r, 400));

                addLog(`Running Sentiment Analysis Model...`, 'info');
                await new Promise(r => setTimeout(r, 800));

                addLog(`Clustering Topics (K-Means)... OK`, 'success');
                addLog(`Finalizing Results...`, 'process');

                await new Promise(r => setTimeout(r, 500));

                renderResults(data);
                resultsContainer.classList.remove('hidden');
                resultsContainer.scrollIntoView({ behavior: 'smooth' });
                showToast(`Berhasil! Ditemukan ${data.total} data.`, 'success');
            } else {
                addLog(`Error: ${data.error || 'Unknown error'}`, 'error');
                showToast('Gagal: ' + (data.error || 'Tidak ada data'), 'error');
            }
        } catch (err) {
            clearInterval(searchInterval);
            addLog(`Connection Failed: ${err.message}`, 'error');
            showToast('Koneksi Gagal: ' + err.message, 'error');
        } finally {
            // Optional: Hide loading after a delay or keep it visible until scrolled?
            // Usually nice to hide it or minimize it. Let's hide it.
            setTimeout(() => {
                loading.classList.add('hidden');
            }, 1000);
        }
    });

    // Train Form Handler
    document.getElementById('train-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const form = e.target;
        const formData = new FormData(form);
        const loading = document.getElementById('train-loading');
        const resultDiv = document.getElementById('train-result');

        loading.classList.remove('hidden');
        resultDiv.classList.add('hidden');
        resultDiv.innerHTML = '';

        try {
            const response = await fetch('/train', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                // Check if started async
                if (data.status === 'started') {
                    showToast(data.message, 'info');
                    pollTrainingStatus();
                } else {
                    // Fallback sync logic (if any)
                    resultDiv.innerHTML = `<div class="alert-success" style="color: green; margin-top: 10px;">${data.message}</div>`;
                    resultDiv.classList.remove('hidden');
                    loading.classList.add('hidden');
                }
            } else {
                resultDiv.innerHTML = `<div class="alert-error" style="color: red; margin-top: 10px;">Error: ${data.error}</div>`;
                showToast('Training Gagal: ' + data.error, 'error');
                resultDiv.classList.remove('hidden');
                loading.classList.add('hidden');
            }
        } catch (err) {
            showToast('Request failed: ' + err.message, 'error');
            loading.classList.add('hidden');
        }
    });
});

function pollTrainingStatus() {
    const loading = document.getElementById('train-loading');
    const resultDiv = document.getElementById('train-result');
    const statusText = loading; // We reuse the loading div text

    let attempts = 0;
    const maxAttempts = 300; // 5 minutes approx (avg 1s)
    let errors = 0;
    const maxErrors = 5;

    const checkStatus = async () => {
        try {
            const res = await fetch('/train_status');
            if (!res.ok) throw new Error("Server error");

            const status = await res.json();
            errors = 0; // Reset errors on success

            if (status.is_training) {
                statusText.textContent = `Sedang melatih model: ${status.message} (${status.progress}%)`;
                loading.classList.remove('hidden');

                attempts++;
                if (attempts > maxAttempts) {
                    showToast("Training memakan waktu terlalu lama. Cek log server/reload nanti.", "warning");
                    return; // Stop polling but leave UI as is or button to retry
                }

                // Exponential polling: slow down if it takes long
                let delay = 1000;
                if (attempts > 60) delay = 5000; // > 1 min

                setTimeout(checkStatus, delay);

            } else {
                loading.classList.add('hidden');

                if (status.result && status.result.success) {
                    resultDiv.innerHTML = `<div class="alert-success" style="color: green; margin-top: 10px;">${status.result.message}</div>`;
                    document.getElementById('model-status').textContent = "Siap Digunakan";
                    document.getElementById('model-status').className = "status-ready";
                    showToast(status.result.message, 'success');
                } else if (status.result && !status.result.success) {
                    resultDiv.innerHTML = `<div class="alert-error" style="color: red; margin-top: 10px;">Error: ${status.result.error}</div>`;
                    showToast('Training Gagal: ' + status.result.error, 'error');
                } else if (!status.result) {
                    // Just idle/finished without result payload (maybe refreshed)
                    // Do nothing
                }

                resultDiv.classList.remove('hidden');
            }
        } catch (e) {
            console.error("Polling error", e);
            errors++;
            if (errors <= maxErrors) {
                setTimeout(checkStatus, 2000 * errors); // Backoff on error
            } else {
                showToast("Gagal mengambil status training berulang kali.", "error");
                loading.classList.add('hidden');
            }
        }
    };

    // Start polling
    checkStatus();
}

let sentimentChart = null;
let clusterChart = null;
let lastResultData = null; // Store data for export

function renderResults(data) {
    lastResultData = data; // Save for export logic

    // Basic Metrics
    document.getElementById('total-reviews').textContent = data.total;

    // Find dominant sentiment
    let dominant = "N/A";
    let maxCount = -1;
    if (data.distribution) {
        for (const [sent, count] of Object.entries(data.distribution)) {
            if (count > maxCount) {
                maxCount = count;
                dominant = sent;
            }
        }
    }
    document.getElementById('dominant-sentiment').textContent = dominant;

    // Charts
    renderSentimentChart(data.distribution);
    renderClusterChart(data.cluster_counts);

    // Helper: Simple HTML Escape
    const escapeHtml = (unsafe) => {
        return (unsafe || "").toString()
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // Table
    const tbody = document.querySelector('#result-table tbody');
    tbody.innerHTML = '';
    data.data.forEach(item => {
        const tr = document.createElement('tr');
        const safeText = escapeHtml(item.text);
        const shortText = safeText.substring(0, 100) + (safeText.length > 100 ? '...' : '');
        const safeCluster = item.cluster !== undefined ? 'Cluster ' + escapeHtml(item.cluster) : '-';
        const safeSentiment = escapeHtml(item.sentiment || '-');

        const safePreprocessed = escapeHtml(item.preprocessed || '-');

        let sourceHtml = '-';
        if (item.source) {
            if (item.source.startsWith('http')) {
                const safeUrl = escapeHtml(item.source);
                const safeTitle = escapeHtml(item.title || item.source);
                sourceHtml = `<a href="${safeUrl}" target="_blank" title="${safeUrl}">${safeTitle}</a>`;
            } else {
                sourceHtml = escapeHtml(item.source);
            }
        }

        tr.innerHTML = `
            <td>${shortText}</td>
            <td style="color: #6b7280; font-style: italic;">${safePreprocessed}</td>
            <td>${sourceHtml}</td>
            <td>${safeSentiment}</td>
            <td>${safeCluster}</td>
        `;
        tbody.appendChild(tr);
    });

    // Initial Warning
    if (!data.distribution || Object.keys(data.distribution).length === 0) {
        document.getElementById('sentiment-warning').classList.remove('hidden');
    } else {
        document.getElementById('sentiment-warning').classList.add('hidden');
    }
}

// Export CSV Function
document.getElementById('download-btn').addEventListener('click', () => {
    if (!lastResultData || !lastResultData.data) {
        showToast("Tidak ada data untuk diunduh.", "error");
        return;
    }

    const rows = [
        ["text", "label", "cluster"] // Header matches training requirements
    ];

    lastResultData.data.forEach(item => {
        // Escape quotes and handle newlines for CSV
        const safeText = `"${item.text.replace(/"/g, '""').replace(/\n/g, ' ')}"`;
        rows.push([safeText, item.sentiment || "", item.cluster !== undefined ? item.cluster : ""]);
    });

    let csvContent = "data:text/csv;charset=utf-8,"
        + rows.map(e => e.join(",")).join("\n");

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    const filename = `dataset_${lastResultData.query || 'export'}_${new Date().toISOString().slice(0, 10)}.csv`;
    link.setAttribute("download", filename);
    document.body.appendChild(link); // Required for FF
    link.click();
    document.body.removeChild(link);

    showToast("Download dimulai!", "success");
});

function renderSentimentChart(distribution) {
    const ctx = document.getElementById('sentimentChart').getContext('2d');

    // Reset canvas to avoid overlap
    if (sentimentChart) sentimentChart.destroy();

    if (!distribution || Object.keys(distribution).length === 0) return;

    sentimentChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: Object.keys(distribution),
            datasets: [{
                data: Object.values(distribution),
                backgroundColor: ['#10b981', '#ef4444', '#f59e0b', '#4f46e5'], // Green, Red, Amber, Indigo
            }]
        }
    });
}

function renderClusterChart(clusterCounts) {
    const ctx = document.getElementById('clusterChart').getContext('2d');

    if (clusterChart) clusterChart.destroy();

    if (!clusterCounts || Object.keys(clusterCounts).length === 0) return;

    const labels = Object.keys(clusterCounts).map(k => `Cluster ${k}`);

    clusterChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Jumlah Ulasan',
                data: Object.values(clusterCounts),
                backgroundColor: '#6366f1'
            }]
        },
        options: {
            scales: {
                y: { beginAtZero: true }
            }
        }
    });
}

function switchTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));

    document.getElementById(`${tabName}-section`).classList.add('active');

    // Find button
    const btns = document.querySelectorAll('.tab-btn');
    // 0: Analyze (Upload), 1: Search, 2: Train
    if (tabName === 'analyze') btns[0].classList.add('active');
    else if (tabName === 'search') btns[1].classList.add('active');
    else if (tabName === 'train') btns[2].classList.add('active');
}
