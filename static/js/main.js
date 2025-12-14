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

        loading.classList.remove('hidden');
        resultsContainer.classList.add('hidden');

        // Indikator "Sedang mencari..."
        showToast(`Mencari data tentang "${query}"...`, 'info');

        try {
            const response = await fetch('/search_and_analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query: query })
            });

            const data = await response.json();

            if (response.ok) {
                renderResults(data);
                resultsContainer.classList.remove('hidden');
                // Scroll to results
                resultsContainer.scrollIntoView({ behavior: 'smooth' });
                showToast(`Berhasil! Ditemukan ${data.total} data.`, 'success');
            } else {
                showToast('Gagal: ' + (data.error || 'Tidak ada data'), 'error');
            }
        } catch (err) {
            showToast('Koneksi Gagal: ' + err.message, 'error');
        } finally {
            loading.classList.add('hidden');
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

        try {
            const response = await fetch('/train', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                resultDiv.innerHTML = `<div class="alert-success" style="color: green; margin-top: 10px;">${data.message} (${data.data_count} data)</div>`;
                document.getElementById('model-status').textContent = "Siap Digunakan";
                document.getElementById('model-status').className = "status-ready";
                showToast(data.message, 'success');
            } else {
                resultDiv.innerHTML = `<div class="alert-error" style="color: red; margin-top: 10px;">Error: ${data.error}</div>`;
                showToast('Training Gagal: ' + data.error, 'error');
            }
            resultDiv.classList.remove('hidden');
        } catch (err) {
            showToast('Request failed: ' + err.message, 'error');
        } finally {
            loading.classList.add('hidden');
        }
    });
});

let sentimentChart = null;
let clusterChart = null;

function renderResults(data) {
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

    // Table
    const tbody = document.querySelector('#result-table tbody');
    tbody.innerHTML = '';
    data.data.forEach(item => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${item.text.substring(0, 100)}${item.text.length > 100 ? '...' : ''}</td>
            <td>${item.sentiment || '-'}</td>
            <td>${item.cluster !== undefined ? 'Cluster ' + item.cluster : '-'}</td>
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
                backgroundColor: ['#10b981', '#ef4444', '#f59e0b', '#3b82f6'], // Adjust as needed
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
