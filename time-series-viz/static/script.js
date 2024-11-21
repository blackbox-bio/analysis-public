document.getElementById('add-graph-btn').addEventListener('click', function() {
    addGraph();
});

function addGraph() {
    const graphId = 'graph-' + Math.random().toString(36).substr(2, 9);
    const container = document.getElementById('graphs-container');

    // Create a container for the graph and dropdown
    const graphContainer = document.createElement('div');
    graphContainer.className = 'graph-container';
    graphContainer.style.display = 'flex';
    graphContainer.style.alignItems = 'center';
    graphContainer.style.marginBottom = '20px';

    // Create the feature selection dropdown
    const featureDropdown = document.createElement('select');
    featureDropdown.className = 'feature-dropdown';
    featureDropdown.style.marginRight = '20px';
    featureDropdown.multiple = true;  // Allow multiple selection

    // Populate the dropdown with features
    fetch('/get_features')
        .then(response => response.json())
        .then(features => {
            features.forEach(feature => {
                const option = document.createElement('option');
                option.value = feature;
                option.text = feature;
                featureDropdown.appendChild(option);
            });
        });

    // Create the graph div
    const graphDiv = document.createElement('div');
    graphDiv.className = 'graph';
    graphDiv.id = graphId;
    graphDiv.style.flex = '1';  // Take up the remaining space

    // Add event listener to the dropdown to update the graph when a feature is selected
    featureDropdown.addEventListener('change', function() {
        const selectedFeatures = Array.from(featureDropdown.selectedOptions).map(option => option.value);
        const sigma = parseFloat(document.getElementById('sigma-input').value);
        updateGraph(graphId, selectedFeatures, sigma);
    });

    // Append the dropdown and graph to the graph container
    graphContainer.appendChild(featureDropdown);
    graphContainer.appendChild(graphDiv);

    // Append the graph container to the main container
    container.appendChild(graphContainer);

    // Initialize the graph with empty data
    Plotly.newPlot(graphId, [], {
        yaxis: { fixedrange: true },
        xaxis: { title: 'Time (frames)' },
        margin: { l: 50, r: 10, t: 25, b: 40 }
    });
}

function updateGraph(graphId, features, sigma) {
    fetch('/get_data', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: features, sigma: sigma })
    })
    .then(response => response.json())
    .then(data => {
        const traces = features.map((feature, index) => {
            const rawData = data[feature]['raw'];
            const smoothedData = data[feature]['smoothed'];

            const xData = Array.from({ length: rawData.length }, (_, i) => i);

            // Plot raw and smoothed data, keeping NaN values to create gaps in the plot
            return [
                {
                    x: xData,
                    y: rawData,
                    mode: 'lines',
                    name: `${feature} - Raw Data`,
                    line: { color: `hsl(${index * 360 / features.length}, 100%, 50%)`, width: 0.5 },
                    opacity: 0.5
                },
                {
                    x: xData,
                    y: smoothedData,
                    mode: 'lines',
                    name: `${feature} - Smoothed Data`,
                    line: { color: `hsl(${index * 360 / features.length}, 100%, 50%)`, width: 2 },
                    opacity: 1
                }
            ];
        }).flat();

        Plotly.react(graphId, traces, {
            yaxis: { fixedrange: true },
            xaxis: { title: 'Time (frames)' },
            margin: { l: 50, r: 10, t: 25, b: 40 }
        });
    });
}
