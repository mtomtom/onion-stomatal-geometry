import dash
from dash import dcc, html, Input, Output, State, ALL, ctx
import plotly.graph_objects as go
import trimesh
import base64
import io
import numpy as np
import os
import tempfile

# Import the NEW refactored analysis functions
from full_length_AR_analysis_dash import calculate_centerline, analyze_sections
from plotting_helpers_dash import get_plotly_figure_for_app, create_section_montage

# --- Initialize the Dash App ---
app = dash.Dash(__name__, prevent_initial_callbacks=True, suppress_callback_exceptions=True)

# --- Define the App Layout ---
app.layout = html.Div([
    html.H1("Interactive Stomata Analysis", style={'textAlign': 'center'}),
    
    # Stores for sharing data between callbacks
    dcc.Store(id='mesh-data-store'),
    dcc.Store(id='centerline-data-store'),
    dcc.Store(id='drawn-seam-store', data=[]),
    dcc.Store(id='montage-data-store'), 

    # --- Main Control Panel ---
    html.Div([
        dcc.Upload(id='upload-mesh', children=html.Button('Upload Mesh File'), style={'display': 'inline-block'}),
        html.Button('Run Initial Analysis', id='run-analysis-button', n_clicks=0, style={'marginLeft': '10px'}),
        dcc.Input(id='num-sections-input', type='number', value=10, min=2, step=1, style={'marginLeft': '20px', 'width': '80px'}),
        html.Label(" Sections", style={'marginLeft': '5px'}),
        # --- ADD DOWNLOAD BUTTON AND COMPONENT ---
        html.Button('Download Montage', id='download-montage-button', n_clicks=0, style={'marginLeft': '20px'}),
        dcc.Download(id="download-montage"),
    ], style={'textAlign': 'center', 'padding': '10px'}),

    # --- Seam Drawing Controls ---
    html.Div([
        html.Button('Start/Stop Drawing Seam', id='draw-seam-button', n_clicks=0),
        html.Button('Clear Drawn Seam', id='clear-seam-button', n_clicks=0),
        html.Div(id='seam-drawing-status', children='Drawing Mode: OFF', style={'display': 'inline-block', 'marginLeft': '20px'}),
    ], style={'textAlign': 'center', 'padding': '5px'}),
    
    # --- Main Display Area ---
    dcc.Loading(id="loading-icon", children=html.Div(id='output-display'), type="circle"),

    # --- Sliders for Fine-Tuning Sections ---
    html.Div(id='sliders-container', style={'padding': '20px'})
])

# --- Callback to handle file upload ---
@app.callback(
    Output('output-display', 'children'),
    Output('mesh-data-store', 'data'),
    Input('upload-mesh', 'contents'),
    State('upload-mesh', 'filename')
)
def handle_file_upload(contents, filename):
    if contents is None: return dash.no_update, dash.no_update

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    mesh = trimesh.load(io.BytesIO(decoded), file_type='obj')
    
    # Use the main plotting function to ensure consistency
    fig = get_plotly_figure_for_app(mesh, None, [])
    
    stored_data = {'contents': content_string, 'filename': filename}
    
    # --- FIX: Give the initial graph the correct ID ---
    return dcc.Graph(id='main-graph', figure=fig, style={'height': '80vh'}), stored_data

# --- FIX 1: Add a new, dedicated callback for the status text ---
@app.callback(
    Output('seam-drawing-status', 'children'),
    Input('draw-seam-button', 'n_clicks')
)
def update_drawing_status(n_clicks):
    return f"Drawing Mode: {'ON' if n_clicks % 2 != 0 else 'OFF'}"

# --- FIX 2: Simplify the seam drawing callback ---
@app.callback(
    Output('drawn-seam-store', 'data'),
    Input('main-graph', 'clickData'),
    Input('clear-seam-button', 'n_clicks'),
    State('draw-seam-button', 'n_clicks'),
    State('drawn-seam-store', 'data')
)
def draw_seam(click_data, clear_clicks, draw_clicks, current_seam_points):
    triggered_id = ctx.triggered_id
    
    if triggered_id == 'clear-seam-button':
        return [] # Only clear the data, don't touch the status
        
    if draw_clicks % 2 != 0 and triggered_id == 'main-graph' and click_data:
        point = click_data['points'][0]
        current_seam_points.append([point['x'], point['y'], point['z']])
        return current_seam_points
        
    return dash.no_update

# --- Main Analysis & Slider Generation Callback ---
@app.callback(
    [Output('centerline-data-store', 'data'),
     Output('sliders-container', 'children'),
     Output('output-display', 'children', allow_duplicate=True),
     Output('montage-data-store', 'data')], # <-- ADD THIS OUTPUT
    [Input('run-analysis-button', 'n_clicks')],
    [State('mesh-data-store', 'data'),
     State('drawn-seam-store', 'data'),
     State('num-sections-input', 'value')]
)
def run_master_analysis(n_clicks, stored_mesh_data, drawn_seam_points, num_sections):
    if not stored_mesh_data: return dash.no_update

    # --- 1. Calculate Centerline (once) ---
    decoded = base64.b64decode(stored_mesh_data['contents'])
    mesh = trimesh.load(io.BytesIO(decoded), file_type='obj')
    
    centerline_data = calculate_centerline(mesh, is_closed=True, seam_points_manual=np.array(drawn_seam_points) if drawn_seam_points else None)
    if not centerline_data:
        return None, "Centerline calculation failed.", html.Div("Error: Could not calculate centerline.")

    # --- 2. Generate Sliders ---
    initial_positions = np.linspace(0, 1, num_sections)
    sliders = [
        html.Div([
            html.Label(f"Section {i+1}:"),
            dcc.Slider(id={'type': 'section-slider', 'index': i}, min=0, max=1, step=0.01, value=pos, marks=None)
        ], key=f"slider-div-{i}") for i, pos in enumerate(initial_positions)
    ]

    # --- 3. Run Initial Section Analysis & Plot ---
    analysis_results = analyze_sections(mesh, centerline_data, initial_positions)
    if not analysis_results: return dash.no_update
    # --- FIX: Use the correct argument name 'centerline_data' ---
    fig = get_plotly_figure_for_app(
        mesh=mesh,
        centerline_data=centerline_data,
        section_data_3d=analysis_results.get('section_data_3d', []),
        seam_points_for_plot=np.array(drawn_seam_points) if drawn_seam_points else centerline_data.get('seam_points_for_plot')
    )
    montage_data = {
        'section_points_list': analysis_results.get('section_points_list', []),
        'positions': analysis_results.get('positions', [])
    }
    return centerline_data, sliders, dcc.Graph(id='main-graph', figure=fig, style={'height': '80vh'}), montage_data

# --- Callback for Updating Plot from Sliders ---
@app.callback(
    [Output('output-display', 'children', allow_duplicate=True),
     Output('montage-data-store', 'data', allow_duplicate=True)], # <-- ADD THIS OUTPUT
    [Input({'type': 'section-slider', 'index': ALL}, 'value')],
    [State('mesh-data-store', 'data'),
     State('centerline-data-store', 'data'),
     State('drawn-seam-store', 'data')]
)
def update_plot_from_sliders(slider_values, stored_mesh_data, centerline_data, drawn_seam_points):
    if not slider_values or not stored_mesh_data or not centerline_data:
        return dash.no_update

    decoded = base64.b64decode(stored_mesh_data['contents'])
    mesh = trimesh.load(io.BytesIO(decoded), file_type='obj')

    analysis_results = analyze_sections(mesh, centerline_data, slider_values)
    if not analysis_results: return dash.no_update
    
    # --- FIX: Use the correct argument name 'centerline_data' ---
    fig = get_plotly_figure_for_app(
        mesh=mesh,
        centerline_data=centerline_data,
        section_data_3d=analysis_results.get('section_data_3d', []),
        seam_points_for_plot=np.array(drawn_seam_points) if drawn_seam_points else centerline_data.get('seam_points_for_plot')
    )
    
    montage_data = {
        'section_points_list': analysis_results.get('section_points_list', []),
        'positions': analysis_results.get('positions', [])
    }
    return dcc.Graph(id='main-graph', figure=fig, style={'height': '80vh'}), montage_data

# --- ADD NEW CALLBACK FOR DOWNLOADING ---
@app.callback(
    Output("download-montage", "data"),
    Input("download-montage-button", "n_clicks"),
    State("montage-data-store", "data"),
    prevent_initial_call=True,
)
def download_montage_figure(n_clicks, montage_data):
    if not montage_data:
        return dash.no_update

    img_bytes = create_section_montage(
        section_points_list=montage_data.get('section_points_list', []),
        positions=montage_data.get('positions', [])
    )
    
    if img_bytes:
        return dcc.send_bytes(img_bytes, "cross_section_montage.png")
    return dash.no_update

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1')



