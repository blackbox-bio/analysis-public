document.getElementById('add-graph-btn').addEventListener('click', function() {
    addGraph();
});

document.getElementById('reset-zoom-btn').addEventListener('click', function() {
    resetZoom();
});

function addGraph() {
    const selectedFeatures = Array.from(document.getElementById('feature-dropdown').selectedOptions).map(option => option.value);
    const sigma = parseFloat(document.getElementById('sigma-input').value);

    if (selectedFeatures.length > 0) {
        fetch('/get_data', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features: selectedFeatures, sigma: sigma })
        })
        .then(response => response.json())
        .then(data => {
            const graphId = 'graph-' + Math.random().toString(36).substr(2, 9);
            const container = document.getElementById('graphs-container');
            const graphDiv = document.createElement('div');
            graphDiv.className = 'graph';
            graphDiv.id = graphId;
            container.appendChild(graphDiv);

            const traces = [];
            selectedFeatures.forEach((feature, index) => {
                const rawTrace = {
                    x: Array.from({length: data[feature].raw.length}, (_, i) => i / {{ recording_fps }}),
                    y: data[feature].raw,
                    mode: 'lines',
                    name: `${feature} - Raw Data`,
                    line: { color: `hsl(${index * 360 / selectedFeatures.length}, 100%, 50%)`, width: 0.5 }
                };
                const smoothedTrace = {
                    x: Array.from({length: data[feature].smoothed.length}, (_, i) => i / {{ recording_fps }}),
                    y: data[feature].smoothed,
                    mode: 'lines',
                    name: `${feature} - Smoothed Data`,
                    line: { color: `hsl(${index * 360 / selectedFeatures.length}, 100%, 50%)`, width: 2 }
                };
                traces.push(rawTrace, smoothedTrace);
            });

            Plotly.newPlot(graphId, traces, {
                yaxis: { fixedrange: true },
                xaxis: { title: 'Time (s)' },
                margin: { l: 50, r: 10, t: 25, b: 40 }
            });
        });
    }
}

function resetZoom() {
    const graphs = document.getElementsByClassName('graph');
    for (let i = 0; i < graphs.length; i++) {
        Plotly.relayout(graphs[i].id, { 'xaxis.range': null, 'yaxis.range': null });
    }
}
