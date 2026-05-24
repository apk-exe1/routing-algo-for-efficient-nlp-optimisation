// DOM Elements
const btnStandard = document.getElementById('btnStandard');
const btnOptimized = document.getElementById('btnOptimized');
const inputText = document.getElementById('inputText');
const outputContainer = document.getElementById('outputContainer');
const analysisContainer = document.getElementById('analysisContainer');

// Metric Elements
const valAlgo = document.getElementById('valAlgo');
const valLatency = document.getElementById('valLatency');
const valWords = document.getElementById('valWords');
const valCache = document.getElementById('valCache');
const valThreads = document.getElementById('valThreads');
const valComplexity = document.getElementById('valComplexity');
const valThroughput = document.getElementById('valThroughput');

// Charts
let threadChart;

const BASE_URL = 'http://localhost:8000';

// Initialize charts
function initCharts() {
    const ctxThread = document.getElementById('threadChart').getContext('2d');

    Chart.defaults.color = '#94a3b8';
    Chart.defaults.font.family = "'Inter', sans-serif";

    threadChart = new Chart(ctxThread, {
        type: 'bar',
        data: {
            labels: ['Thread 1'],
            datasets: [{
                label: 'Workload (%)',
                data: [0],
                backgroundColor: ['#a855f7'],
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: { color: 'rgba(30, 41, 59, 0.5)' }
                },
                x: {
                    grid: { display: false }
                }
            },
            plugins: { 
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.parsed.y + '% Workload';
                        }
                    }
                }
            }
        }
    });
}

// Global state for chart updates
let lastStandardLatency = 0;
let lastOptimizedLatency = 0;

function updateCharts(algo, latency, moduleBreakdown, threadUtilization) {
    
    // Custom HTML Module Performance Breakdown
    const colors = ['#3b82f6', '#22c55e', '#f97316', '#a855f7', '#06b6d4', '#eab308', '#ef4444', '#d946ef', '#84cc16'];
    const emojis = ['📝', '🔄', '🛡️', '⚙️', '🔍', '⏳', '🧩', '🏷️', '📊'];

    // Sort modules by percentage
    const sortedModules = Object.entries(moduleBreakdown)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 8); // Top 8 modules
        
    let html = `<div style="display: flex; flex-direction: column; gap: 12px;">`;
    
    sortedModules.forEach((m, index) => {
        const label = m[0].replace('.py', '').replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()); // Convert to title case
        const percent = m[1].toFixed(2);
        const color = colors[index % colors.length];
        const emoji = emojis[index % emojis.length];
        
        html += `
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div style="display: flex; align-items: center; width: 45%; gap: 10px;">
                    <span style="font-size: 1.2em;">${emoji}</span>
                    <span style="color: #e2e8f0; font-size: 0.9em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${label}</span>
                </div>
                <div style="flex-grow: 1; margin: 0 15px; background: rgba(30, 41, 59, 0.5); height: 12px; border-radius: 6px; overflow: hidden; display: flex; align-items: center;">
                    <div style="width: ${percent}%; height: 100%; background-color: ${color}; border-radius: 6px;"></div>
                </div>
                <div style="width: 15%; text-align: right; color: #e2e8f0; font-size: 0.9em;">
                    ${percent}%
                </div>
            </div>
        `;
    });
    
    // Add x-axis
    html += `
        <div style="display: flex; justify-content: space-between; padding-left: 45%; padding-right: 15%; margin-top: 10px; border-top: 1px solid rgba(148, 163, 184, 0.2); padding-top: 5px;">
            <span style="color: #94a3b8; font-size: 0.8em;">0%</span>
            <span style="color: #94a3b8; font-size: 0.8em;">20%</span>
            <span style="color: #94a3b8; font-size: 0.8em;">40%</span>
            <span style="color: #94a3b8; font-size: 0.8em;">60%</span>
            <span style="color: #94a3b8; font-size: 0.8em;">80%</span>
            <span style="color: #94a3b8; font-size: 0.8em;">100%</span>
        </div>
        <div style="text-align: center; color: #94a3b8; font-size: 0.8em; margin-top: 5px;">Utilization (%)</div>
    `;
    
    html += `</div>`;
    document.getElementById('modulePerformanceContainer').innerHTML = html;
    
    
    // Update Thread Chart
    const threadLabels = threadUtilization.map((t, i) => `Thread ${i + 1}`);
    const threadData = threadUtilization;
    const threadColors = ['#a855f7', '#06b6d4', '#eab308', '#ef4444', '#3b82f6'];
    
    threadChart.data.labels = threadLabels;
    threadChart.data.datasets[0].data = threadData;
    threadChart.data.datasets[0].backgroundColor = threadLabels.map((_, i) => threadColors[i % threadColors.length]);
    
    threadChart.update();
}

// Set Loading State
function setLoading(isStandard) {
    outputContainer.innerHTML = '<span class="placeholder">Processing...</span>';
    outputContainer.classList.remove('success');
    analysisContainer.innerHTML = '<span class="placeholder">Analyzing text...</span>';
    
    valAlgo.innerText = isStandard ? 'Linear Search' : 'BK-Trees + LRU + Parallel';
    valLatency.innerText = '...';
    valWords.innerText = '...';
    valCache.innerText = '...';
    valThreads.innerText = '...';
    valComplexity.innerText = '...';
    valThroughput.innerText = '...';

    if (isStandard) {
        btnStandard.disabled = true;
        btnOptimized.disabled = true;
    } else {
        btnStandard.disabled = true;
        btnOptimized.disabled = true;
    }
}

// Reset Loading State
function resetLoading() {
    btnStandard.disabled = false;
    btnOptimized.disabled = false;
}

// Handle API Call
async function runAlgorithm(endpoint, isStandard) {
    const text = inputText.value.trim();
    if (!text) {
        alert("Please enter some text");
        return;
    }

    setLoading(isStandard);

    try {
        const response = await fetch(`${BASE_URL}/${endpoint}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text })
        });

        if (!response.ok) throw new Error("Network response was not ok");

        const data = await response.json();
        
        // Update UI
        outputContainer.innerHTML = data.corrected_text;
        outputContainer.classList.add('success');
        
        valAlgo.innerText = isStandard ? 'Linear Search' : 'BK-Trees + LRU + Parallel';
        valLatency.innerText = `${data.latency_ms} ms`;
        valWords.innerText = data.words_processed;
        valCache.innerText = data.cache_hits;
        valThreads.innerText = data.parallel_threads;
        valComplexity.innerText = data.time_complexity;
        valThroughput.innerText = data.throughput;

        // Word analysis display
        let analysisHtml = `<div class="analysis-content">
            <p><strong>Processed ${data.words_processed} words.</strong></p>
            <p>Algorithm: ${data.algorithm}</p>
            <p>Time Complexity: ${data.time_complexity}</p>
            <p>Threads utilized: ${data.parallel_threads}</p>`;
            
        if (!isStandard) {
            analysisHtml += `<p>BK-Tree Search enabled.</p>
            <p>LRU Cache Hits: ${data.cache_hits}</p>`;
        } else {
            analysisHtml += `<p>Linear search performed over full dictionary space.</p>`;
        }
        // Removed text module list
        if (data.word_analysis && data.word_analysis.length > 0) {
            analysisHtml += `<br><p><strong>Detailed Word Analysis:</strong></p><div style="max-height: 200px; overflow-y: auto; background-color: #0b0f19; padding: 10px; border-radius: 4px; font-size: 0.85em;">`;
            data.word_analysis.forEach(wa => {
                const cacheInfo = wa.cache_hit ? `<span style="color: #22c55e;">[CACHE HIT]</span> ` : "";
                analysisHtml += `<div style="margin-bottom: 8px; border-bottom: 1px solid #1e293b; padding-bottom: 4px;">
                    <div><span style="color: #94a3b8;">Original:</span> ${wa.original}</div>
                    <div><span style="color: #94a3b8;">Corrected:</span> <span style="color: #3b82f6;">${wa.corrected}</span></div>
                    <div><span style="color: #94a3b8;">Latency:</span> ${wa.latency_ms} ms ${cacheInfo}</div>
                </div>`;
            });
            analysisHtml += `</div>`;
        }
        
        analysisHtml += `</div>`;
        analysisContainer.innerHTML = analysisHtml;

        // Update visual charts
        updateCharts(data.algorithm, data.latency_ms, data.module_breakdown, data.thread_utilization);

    } catch (error) {
        console.error("Error:", error);
        outputContainer.innerHTML = `<span class="placeholder" style="color: #ef4444;">Error connecting to backend server. Make sure FastAPI is running.</span>`;
        analysisContainer.innerHTML = `<span class="placeholder">Failed.</span>`;
    } finally {
        resetLoading();
    }
}

// Event Listeners
btnStandard.addEventListener('click', () => runAlgorithm('run_standard', true));
btnOptimized.addEventListener('click', () => runAlgorithm('run_optimized', false));

// Init
document.addEventListener('DOMContentLoaded', initCharts);
