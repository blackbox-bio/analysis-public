import os
import h5py
import numpy as np
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State, ALL
import plotly.graph_objs as go
from scipy.ndimage import gaussian_filter1d
import uuid

# Define project directory and file paths
proj_dir = r"D:\data\TAMU\hallway-tests_analysis"
# proj_dir = r"D:\data\Twiss_lab\test_analysis"
recording_name = "2024-07-31_15-19-11_7917-1-sec-dwell-training-sci"
# recording_name = "0014"
features_h5 = os.path.join(proj_dir, recording_name, "features.h5")

# Load the HDF5 file and extract the relevant information into a dictionary for faster access
features_data = {}
recording_fps = 45  # Assuming a fixed FPS for now
with h5py.File(features_h5, "r") as features_df:
    recording_id = list(features_df.keys())[0]
    feature_cols = list(features_df[recording_id].keys())
    for feature in feature_cols:
        if features_df[recording_id][feature].shape == ():  # Scalar
            features_data[feature] = features_df[recording_id][feature][()]
        else:  # Non-scalar, it's an array
            features_data[feature] = features_df[recording_id][feature][:]
    recording_fps = features_df[recording_id]["fps"][()]
    recording_frame_count = features_df[recording_id]["frame_count"][()]
    feature_cols = [col for col in feature_cols if col not in ["fps", "frame_count"]]

# Initialize the Dash app with a Bootstrap theme and suppress callback exceptions
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Define fixed dimensions for each graph
fixed_graph_height = 300
fixed_graph_width = 1200

# Color palette for better differentiation
color_palette = [
    'rgb(31, 119, 180)',  # Blue
    'rgb(255, 127, 14)',  # Orange
    'rgb(44, 160, 44)',   # Green
    'rgb(214, 39, 40)',   # Red
    'rgb(148, 103, 189)', # Purple
    'rgb(140, 86, 75)',   # Brown
    'rgb(227, 119, 194)', # Pink
    'rgb(127, 127, 127)', # Gray
    'rgb(188, 189, 34)',  # Yellow-green
    'rgb(23, 190, 207)'   # Cyan
]

# Function to convert frames to time in 00:00:00.x format (one decimal for milliseconds)
def frames_to_time(frames, fps, show_milliseconds=False):
    seconds = frames / fps
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = round((frames / fps - int(frames / fps)) * 10, 1)
    if show_milliseconds:
        if hours > 0:
            return f"{hours:02}:{minutes:02}:{seconds:02}.{int(milliseconds)}"
        elif minutes > 0:
            return f"{minutes:02}:{seconds:02}.{int(milliseconds)}"
        else:
            return f"{seconds:02}.{int(milliseconds)}"
    else:
        if hours > 0:
            return f"{hours:02}:{minutes:02}:{seconds:02}"
        elif minutes > 0:
            return f"{minutes:02}:{seconds:02}"
        else:
            return f"{seconds:02}"

def add_line_breaks(text, char_limit=15):
    """Insert line breaks into the text at every `char_limit` characters, even within words."""
    new_text = ""
    while len(text) > char_limit:
        new_text += text[:char_limit] + "<br>"
        text = text[char_limit:]
    new_text += text
    return new_text

# Initial figure for the hidden timeline graph
initial_timeline_figure = go.Figure(
    layout=go.Layout(
        xaxis=dict(range=[0, recording_frame_count / recording_fps])
    )
)

# Layout of the Dash app
app.layout = dbc.Container([
    dcc.Store(id='x-axis-range-store'),  # Store for the x-axis range
    dbc.Row([
        dbc.Col([
            html.Button('Add Graph', id='add-graph-btn', n_clicks=0),
            html.Button('Reset Zoom', id='reset-zoom-btn', n_clicks=0, style={'margin-left': '10px'}),
            html.Br(),
            html.Label(f'Sigma (Multiplier of FPS = {recording_fps}):'),
            dcc.Input(
                id='sigma-input',
                type='number',
                value=1.0,  # Default to 1 times fps
                min=0.1,
                step=0.1,
                style={'width': '100px'},
                debounce=True
            ),
            html.Br(),
            html.Label('Select Feature to Plot:'),
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col(html.Div(id='graphs-container'), width=12)
    ]),
    html.Div(dcc.Graph(id='hidden-timeline', style={'display': 'none'}, figure=initial_timeline_figure))  # Hidden timeline graph
], fluid=True, style={'height': '100vh', 'overflowY': 'auto', 'padding': '10px'})

# Callback to add an empty graph and keep them in sync
@app.callback(
    Output('graphs-container', 'children'),
    [Input('add-graph-btn', 'n_clicks')],
    [State('graphs-container', 'children')]
)
def add_empty_graph(n_clicks, existing_graphs):
    if n_clicks == 0:
        return existing_graphs

    if existing_graphs is None:
        existing_graphs = []

    new_graph_id = str(uuid.uuid4())

    empty_graph = dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id={'type': 'feature-dropdown', 'index': new_graph_id},
                options=[{'label': col, 'value': col} for col in feature_cols],
                value=[],
                multi=True,
                placeholder="Select feature(s) to plot..."
            ),
            dcc.Checklist(
                id={'type': 'hide-raw-checkbox', 'index': new_graph_id},
                options=[{'label': 'Hide Raw Trace', 'value': 'hide'}],
                value=[],
                inline=True
            )
        ], width=2, style={'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center'}),
        dbc.Col([
            dcc.Graph(
                id={'type': 'dynamic-graph', 'index': new_graph_id},
                config={
                    'scrollZoom': False,  # Disable scroll zoom for individual graphs
                    'doubleClick': 'reset+autosize'  # Disable double-click reset
                },
                figure=go.Figure(
                    layout=go.Layout(
                        yaxis_title='Signal',
                        showlegend=True,
                        height=fixed_graph_height,
                        width=fixed_graph_width,
                        margin=dict(l=40, r=20, t=40, b=20),
                        yaxis=dict(fixedrange=True),
                        xaxis=dict(range=[0, recording_frame_count / recording_fps], showticklabels=False),
                        legend=dict(
                            orientation="v",  # Vertical orientation
                            x=1.02,  # Position the legend on the right
                            y=1,
                            xanchor="left",
                            yanchor="top",
                            bgcolor='rgba(255,255,255,0.7)',
                            bordercolor='rgba(0,0,0,0)',
                            borderwidth=0,
                            itemsizing='constant',
                            title=dict(text='', side='top'),
                        ),
                        dragmode='zoom',
                    )
                ),
                style={'height': f'{fixed_graph_height}px', 'width': f'{fixed_graph_width}px'}
            )
        ], width=10)
    ], style={'marginBottom': '0px'})

    existing_graphs.append(empty_graph)

    return existing_graphs

# Combined callback to update the x-axis range store
@app.callback(
    Output('x-axis-range-store', 'data'),
    [Input({'type': 'dynamic-graph', 'index': ALL}, 'relayoutData'),
     Input('reset-zoom-btn', 'n_clicks')],
    [State('x-axis-range-store', 'data')],
    prevent_initial_call=True
)
def update_x_axis_range(relayout_data_list, reset_clicks, current_range):
    ctx = dash.callback_context

    if not ctx.triggered:
        return current_range

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'reset-zoom-btn':
        # Reset to the full range
        return {
            'x_min': 0,
            'x_max': recording_frame_count / recording_fps
        }

    for relayout_data in relayout_data_list:
        if relayout_data and 'xaxis.range[0]' in relayout_data:
            return {
                'x_min': relayout_data['xaxis.range[0]'],
                'x_max': relayout_data['xaxis.range[1]']
            }

    return current_range

# Callback to synchronize all graphs based on the x-axis range store
@app.callback(
    Output({'type': 'dynamic-graph', 'index': ALL}, 'figure'),
    [Input('x-axis-range-store', 'data'),
     Input({'type': 'feature-dropdown', 'index': ALL}, 'value')],
    [State('sigma-input', 'value')]
)
def sync_graphs_with_x_axis_range(x_axis_range, selected_features_list, sigma_multiplier):
    sigma = sigma_multiplier * recording_fps  # Convert multiplier to actual sigma
    updated_figures = []

    # Ensure we have an updated figure for each graph
    for selected_features in selected_features_list:
        if x_axis_range:
            x_min = x_axis_range['x_min']
            x_max = x_axis_range['x_max']
        else:
            x_min, x_max = 0, recording_frame_count / recording_fps

        x_range = np.arange(0, recording_frame_count) / recording_fps
        x_range = x_range[(x_range >= x_min) & (x_range <= x_max)]
        tick_labels = [frames_to_time(frame * recording_fps, recording_fps) for frame in x_range]

        figure_data = []
        y_min, y_max = None, None

        for j, feature in enumerate(selected_features):
            data = features_data[feature]  # Access pre-loaded data
            smoothed_data = gaussian_filter1d(data, sigma=sigma)
            color = color_palette[j % len(color_palette)]

            legend_label_raw = f'{feature} - Raw Data'
            legend_label_smoothed = f'{feature} - Smoothed Data'
            legend_label_raw = add_line_breaks(legend_label_raw, char_limit=20)
            legend_label_smoothed = add_line_breaks(legend_label_smoothed, char_limit=20)

            if y_min is None or y_max is None:
                y_min, y_max = np.min(smoothed_data), np.max(smoothed_data)
            else:
                y_min = min(y_min, np.min(smoothed_data))
                y_max = max(y_max, np.max(smoothed_data))

            figure_data.extend([
                go.Scatter(x=x_range, y=data[(x_range * recording_fps).astype(int)], mode='lines', name=legend_label_raw,
                           line=dict(color=color, width=0.5), opacity=0.5),
                go.Scatter(x=x_range, y=smoothed_data[(x_range * recording_fps).astype(int)], mode='lines', name=legend_label_smoothed,
                           line=dict(color=color, width=2), opacity=1)
            ])

        layout_update = {
            'yaxis': dict(fixedrange=True, range=[y_min, y_max]),
            'xaxis': dict(
                range=[x_min, x_max],  # Apply the zoomed range from the timeline
                tickvals=x_range[::len(x_range)//10] if len(x_range) > 10 else x_range,
                ticktext=tick_labels[::len(tick_labels)//10] if len(tick_labels) > 10 else tick_labels,
                tickmode='array',
            ),
            'legend': dict(
                orientation="v",  # Vertical orientation
                x=1.02,  # Position the legend on the right
                y=1,
                xanchor="left",
                yanchor="top",
                bgcolor='rgba(255,255,255,0.7)',
                bordercolor='rgba(0,0,0,0)',
                borderwidth=0,
                itemsizing='constant',
                title=dict(text='', side='top'),
            ),
            'dragmode': 'zoom',
            'margin': dict(l=40, r=20, t=40, b=20)
        }

        updated_figures.append({'data': figure_data, 'layout': layout_update})

    return updated_figures

if __name__ == '__main__':
    app.run_server(debug=True)
