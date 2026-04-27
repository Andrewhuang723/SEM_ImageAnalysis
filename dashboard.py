import os

import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import io
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage import filters, morphology, measure, feature, segmentation, color
from skimage.segmentation import clear_border
from datetime import datetime

# Initialize Dash app
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# Global variables to store data
original_image_data = None  # Store original uploaded image
image_data = None           # Store current working image (cropped or original)
binary_data = None
labeled_data = None
regions_data = None
analysis_log = []           # Store log of all analyzed images
current_filename = None     # Store current image filename

def parse_contents(contents, filename=None):
    """Parse uploaded image contents to numpy array"""
    global current_filename
    
    if contents is None:
        return None
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    # Load image using PIL
    image = Image.open(io.BytesIO(decoded))
    
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Store current filename
    if filename:
        current_filename = filename
    else:
        current_filename = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return np.array(image)

def extract_filename_from_upload(contents, filename):
    """Extract filename from upload component"""
    if filename:
        return filename
    return f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def crop_image(image, top, bottom, left, right):
    """Crop image based on provided coordinates"""
    if image is None:
        return None
    
    height, width = image.shape
    
    # Handle negative values (from end)
    if bottom == -1 or bottom >= height:
        bottom = height
    if right == -1 or right >= width:
        right = width
    
    # Validate crop boundaries
    top = max(0, min(top, height-1))
    bottom = max(top+1, min(bottom, height))
    left = max(0, min(left, width-1))
    right = max(left+1, min(right, width))
    
    return image[top:bottom, left:right]

def create_binary_image(image, threshold=None, threshold_2=None):
    """Create binary image based on thresholds"""
    if image is None:
        return None

    if threshold is None:
        threshold = filters.threshold_otsu(image)

    if threshold_2 is not None:
        binary_image = (image < threshold) & (image > threshold_2)
    else:
        binary_image = image < threshold
    return binary_image


def analyze_regions(binary_img, original_img):
    """Analyze dark regions in binary image"""
    if binary_img is None:
        return None, None
    
    # Clean up the binary image
    cleaned = morphology.remove_small_objects(binary_img, min_size=50)
    cleaned = clear_border(cleaned)
    
    # Label connected components
    labeled = measure.label(cleaned)
    
    # Get region properties
    props = measure.regionprops(labeled, intensity_image=original_img)
    
    return labeled, props


def anaylyze_particle_regions(binary_img, original_img):
    """Analyze dark regions in binary image"""
    binary = ndi.binary_fill_holes(binary_img)
    binary = morphology.binary_opening(binary, footprint=morphology.disk(3))
    distance = ndi.distance_transform_edt(binary)
    coords = feature.peak_local_max(distance, min_distance=5, labels=binary)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labeled = segmentation.watershed(-distance, markers, mask=binary)
    props = measure.regionprops(labeled, intensity_image=original_img)
    
    return labeled, props

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("SEM Image Analysis Dashboard", className="text-center mb-4"),
            html.Hr()
        ])
    ]),
    
    dcc.Tabs([
        dcc.Tab(label="Analysis", children=[
            # Block 1: Image Upload and Display
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Block 1: Upload SEM Image & Crop")),
                        dbc.CardBody([
                            dcc.Upload(
                                id='upload-image',
                                children=html.Div([
                                    'Drag and Drop or ',
                                    html.A('Select SEM Image')
                                ]),
                                style={
                                    'width': '100%',
                                    'height': '60px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin': '10px'
                                },
                                multiple=False
                            ),
                            html.Br(),
                            # Cropping controls
                            html.Div(id='crop-controls', children=[
                                html.Label("Crop Area (pixels):"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Top:"),
                                        dcc.Input(id='crop-top', type='number', value=0, min=0)
                                    ], width=3),
                                    dbc.Col([
                                        html.Label("Bottom:"),
                                        dcc.Input(id='crop-bottom', type='number', value=-1, min=-1)
                                    ], width=3),
                                    dbc.Col([
                                        html.Label("Left:"),
                                        dcc.Input(id='crop-left', type='number', value=0, min=0)
                                    ], width=3),
                                    dbc.Col([
                                        html.Label("Right:"),
                                        dcc.Input(id='crop-right', type='number', value=-1, min=-1)
                                    ], width=3)
                                ]),
                                html.Br(),
                                dbc.Button("Apply Crop", id='apply-crop-btn', color='primary', size='sm'),
                                dbc.Button("Reset", id='reset-crop-btn', color='secondary', size='sm', className='ms-2')
                            ], style={'display': 'none'}),
                            html.Br(),
                            dcc.Graph(id='original-image')
                        ])
                    ])
                ], width=6),
                
                # Block 2: Threshold Control and Binary Image
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Block 2: Threshold Control")),
                        dbc.CardBody([
                            html.Label("Binary Mode:"),
                            dcc.Dropdown(
                                id='binary-mode',
                                options=[
                                    {'label': 'Particle', 'value': 'particle'},
                                    {'label': 'Pore', 'value': 'pore'}
                                ],
                                value='pore',
                                clearable=False
                            ),
                            html.Br(),
                            html.Label("Threshold Value:"),
                            dcc.Slider(
                                id='threshold-slider',
                                min=0,
                                max=255,
                                step=1,
                                value=127,
                                marks={i: str(i) for i in range(0, 256, 50)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            html.Br(),
                            html.Label("Threshold 2 (Optional):"),
                            dcc.Slider(
                                id='threshold-2-slider',
                                min=0,
                                max=255,
                                step=1,
                                value=0,
                                marks={i: str(i) for i in range(0, 256, 50)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            html.Br(),
                            dcc.Graph(id='binary-image')
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Block 4: Distribution Analysis
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Block 4: Area Distribution")),
                        dbc.CardBody([
                            dcc.Graph(id='distribution-plot'),
                            html.Div(id='distribution-stats')
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Save Analysis Button Row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                dbc.Button(
                                    "Save Current Analysis", 
                                    id='save-analysis-btn', 
                                    color='success', 
                                    size='lg', 
                                    className='me-3'
                                ),
                                dbc.Button(
                                    "Clear All Logs", 
                                    id='clear-log-btn', 
                                    color='danger', 
                                    size='lg',
                                    outline=True
                                ),
                                dbc.Button(
                                    "Export to CSV", 
                                    id='export-csv-btn', 
                                    color='info', 
                                    size='lg',
                                    className='ms-3'
                                ),
                                dcc.Download(id="download-csv")
                            ], className='text-center'),
                            html.Div(id='save-status', className='mt-3')
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Block 5: Analysis Log
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Block 5: Analysis Log")),
                        dbc.CardBody([
                            html.Div([
                                html.H6("Analysis Summary:", className="mb-3"),
                                html.Div(id='log-summary'),
                                html.Hr(),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Pixel-to-µm scale (µm/pixel):"),
                                        dcc.Input(
                                            id='scale-factor',
                                            type='number',
                                            min=0,
                                            step=0.001,
                                            placeholder='e.g., 0.05'
                                        )
                                    ], width=6),
                                    dbc.Col([
                                        html.Div("Leave empty to keep pixels.", className='text-muted')
                                    ], width=6)
                                ], className="mb-3"),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Button(
                                            "Delete Selected Rows", 
                                            id='delete-selected-btn', 
                                            color='warning', 
                                            size='sm',
                                            disabled=True
                                        )
                                    ], width=6),
                                    dbc.Col([
                                        html.Div(id='selection-info', className='text-muted')
                                    ], width=6)
                                ], className="mb-3"),
                                dash_table.DataTable(
                                    id='analysis-log-table',
                                    columns=[
                                        {'name': 'Timestamp', 'id': 'timestamp'},
                                        {'name': 'Filename', 'id': 'filename'},
                                        {'name': 'Threshold', 'id': 'threshold', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                        {'name': 'Threshold 2', 'id': 'threshold_2', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                        {'name': 'Binary Mode', 'id': 'binary_mode'},
                                        {'name': 'Scale (µm/px)', 'id': 'scale_factor', 'type': 'numeric', 'format': {'specifier': '.4f'}},
                                        {'name': 'D10', 'id': 'd10', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                        {'name': 'D50', 'id': 'd50', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                        {'name': 'D90', 'id': 'd90', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                        {'name': 'Porosity (%)', 'id': 'porosity', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                        {'name': 'Pore Count', 'id': 'pore_count', 'type': 'numeric'},
                                        {'name': 'Mean Pore Size (px)', 'id': 'mean_pore_size', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                                        {'name': 'Total Dark Area (px)', 'id': 'total_dark_area', 'type': 'numeric'}
                                    ],
                                    data=[],
                                    row_selectable="multi",
                                    selected_rows=[],
                                    style_table={'overflowX': 'auto'},
                                    style_cell={
                                        'textAlign': 'center',
                                        'padding': '10px',
                                        'fontFamily': 'Arial'
                                    },
                                    style_header={
                                        'backgroundColor': 'rgb(230, 230, 230)',
                                        'fontWeight': 'bold'
                                    },
                                    style_data_conditional=[
                                        {
                                            'if': {'row_index': 'odd'},
                                            'backgroundColor': 'rgb(248, 248, 248)'
                                        },
                                        {
                                            'if': {'state': 'selected'},
                                            'backgroundColor': 'rgba(255, 193, 7, 0.2)',
                                            'border': '1px solid rgb(255, 193, 7)'
                                        }
                                    ],
                                    sort_action="native",
                                    page_action="native",
                                    page_size=10
                                )
                            ])
                        ])
                    ])
                ])
            ])
        ]),
        dcc.Tab(label="Diameter Distribution", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Particle Diameter Distribution")),
                        dbc.CardBody([
                            dcc.Graph(id='diameter-plot'),
                            html.Div(id='diameter-stats')
                        ])
                    ])
                ], width=12)
            ], className="mb-4")
        ])
    ])
], fluid=True)

# Callback for showing crop controls when image is uploaded
@app.callback(
    Output('crop-controls', 'style'),
    [Input('upload-image', 'contents')]
)
def show_crop_controls(contents):
    if contents is None:
        return {'display': 'none'}
    return {'display': 'block'}

# Callback to update crop input max values based on image dimensions
@app.callback(
    [Output('crop-top', 'max'),
     Output('crop-bottom', 'max'),
     Output('crop-left', 'max'),
     Output('crop-right', 'max'),
     Output('crop-bottom', 'value'),
     Output('crop-right', 'value')],
    [Input('upload-image', 'contents'),
     Input('upload-image', 'filename')]
)
def update_crop_limits(contents, filename):
    if contents is None:
        return 0, 0, 0, 0, -1, -1
    
    global original_image_data, current_filename
    original_image_data = parse_contents(contents, filename)
    
    if original_image_data is None:
        return 0, 0, 0, 0, -1, -1
    
    height, width = original_image_data.shape
    return height-1, height, width-1, width, height, width

# Callback for image upload and display
@app.callback(
    Output('original-image', 'figure'),
    [Input('upload-image', 'contents'),
     Input('apply-crop-btn', 'n_clicks'),
     Input('reset-crop-btn', 'n_clicks')],
    [State('crop-top', 'value'),
     State('crop-bottom', 'value'),
     State('crop-left', 'value'),
     State('crop-right', 'value')]
)
def update_original_image(contents, apply_clicks, reset_clicks, crop_top, crop_bottom, crop_left, crop_right):
    global original_image_data, image_data
    
    if contents is None:
        return px.scatter(title="Upload an image to begin analysis")
    
    # Determine which button was clicked
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    # Parse original image if not already done
    if original_image_data is None:
        original_image_data = parse_contents(contents)
    
    if original_image_data is None:
        return px.scatter(title="Error loading image")
    
    # Decide whether to use original or cropped image
    if trigger_id == 'reset-crop-btn' or trigger_id == 'upload-image':
        # Use original image
        image_data = original_image_data.copy()
        title_suffix = "Original SEM Image"
    elif trigger_id == 'apply-crop-btn' and all(v is not None for v in [crop_top, crop_bottom, crop_left, crop_right]):
        # Apply cropping
        image_data = crop_image(original_image_data, crop_top, crop_bottom, crop_left, crop_right)
        title_suffix = f"Cropped SEM Image ({crop_top}:{crop_bottom}, {crop_left}:{crop_right})"
    else:
        # Default to original
        image_data = original_image_data.copy()
        title_suffix = "Original SEM Image"
    
    if image_data is None:
        return px.scatter(title="Error processing image")
    
    # Create figure
    fig = px.imshow(image_data, color_continuous_scale='gray', title=title_suffix)
    fig.update_layout(
        xaxis_title="Pixels",
        yaxis_title="Pixels",
        coloraxis_showscale=True
    )
    
    return fig

# Callback for threshold control and binary image
@app.callback(
    Output('binary-image', 'figure'),
    [Input('threshold-slider', 'value'),
    Input('threshold-2-slider', 'value'),
    Input('binary-mode', 'value'),
     Input('upload-image', 'contents'),
     Input('apply-crop-btn', 'n_clicks'),
     Input('reset-crop-btn', 'n_clicks')]
)
def update_binary_image(threshold, threshold_2, binary_mode, contents, apply_clicks, reset_clicks):
    global binary_data
    
    if image_data is None:
        return px.scatter(title="Upload an image first")
    
    threshold_2_value = threshold_2 if threshold_2 and threshold_2 > 0 else None

    # Create binary image
    binary_data = create_binary_image(image_data, threshold, threshold_2_value)
    
    # Calculate porosity
    porosity = np.sum(binary_data) / binary_data.size * 100

    if binary_mode == 'particle':
        labeled_data, regions_data = anaylyze_particle_regions(binary_data, image_data)
    else:
        labeled_data, regions_data = analyze_regions(binary_data, image_data)
    
    # # Create figure (label overlay on original image)
    # labels = measure.label(binary_data)
    overlay = color.label2rgb(labeled_data, image=image_data, bg_label=0)
    fig = px.imshow(
        overlay,
        title=f"Binary Image (Threshold: {threshold}, Porosity: {porosity:.1f}%)"
    )
    fig.update_layout(
        xaxis_title="Pixels",
        yaxis_title="Pixels"
    )
    
    return fig

# Callback for distribution plot
@app.callback(
    [Output('distribution-plot', 'figure'),
     Output('distribution-stats', 'children')],
    [Input('threshold-slider', 'value'),
    Input('threshold-2-slider', 'value'),
    Input('binary-mode', 'value'),
     Input('upload-image', 'contents'),
     Input('apply-crop-btn', 'n_clicks'),
     Input('reset-crop-btn', 'n_clicks')]
)
def update_distribution(threshold, threshold_2, binary_mode, contents, apply_clicks, reset_clicks):
    global labeled_data, regions_data
    
    if image_data is None or binary_data is None:
        return px.scatter(title="Process image first"), ""
    
    # Analyze regions based on mode
    if binary_mode == 'particle':
        labeled_data, regions_data = anaylyze_particle_regions(binary_data, image_data)
    else:
        labeled_data, regions_data = analyze_regions(binary_data, image_data)
    
    if regions_data is None or len(regions_data) == 0:
        return px.scatter(title="No regions to analyze"), ""
    
    # Extract areas
    areas = [region.area for region in regions_data]
    diameters = [region.equivalent_diameter_area for region in regions_data]
    
    # Create histogram
    fig = px.histogram(
        x=areas,
        nbins=30,
        title="Distribution of Dark Region Areas",
        labels={'x': 'Area (pixels)', 'y': 'Count'}
    )
    
    # Add statistics line
    mean_area = np.mean(areas)
    fig.add_vline(x=mean_area, line_dash="dash", line_color="red", 
                  annotation_text=f"Mean: {mean_area:.1f}")
    
    fig.update_layout(
        xaxis_title="Area (pixels)",
        yaxis_title="Frequency",
        showlegend=False
    )
    
    # Statistics
    stats = html.Div([
        html.H6("Distribution Statistics:"),
        html.P(f"Mean area: {np.mean(areas):.1f} pixels"),
        html.P(f"Median area: {np.median(areas):.1f} pixels"),
        html.P(f"Std deviation: {np.std(areas):.1f} pixels"),
        html.P(f"Min area: {np.min(areas):.0f} pixels"),
        html.P(f"Max area: {np.max(areas):.0f} pixels")
    ])
    
    return fig, stats

# Callback for diameter distribution page
@app.callback(
    [Output('diameter-plot', 'figure'),
     Output('diameter-stats', 'children')],
    [Input('threshold-slider', 'value'),
     Input('threshold-2-slider', 'value'),
     Input('binary-mode', 'value'),
     Input('upload-image', 'contents'),
     Input('apply-crop-btn', 'n_clicks'),
     Input('reset-crop-btn', 'n_clicks')]
)
def update_diameter_distribution(threshold, threshold_2, binary_mode, contents, apply_clicks, reset_clicks):
    global labeled_data, regions_data

    if image_data is None or binary_data is None:
        return px.scatter(title="Process image first"), ""

    if binary_mode == 'particle':
        labeled_data, regions_data = anaylyze_particle_regions(binary_data, image_data)
    else:
        labeled_data, regions_data = analyze_regions(binary_data, image_data)

    if regions_data is None or len(regions_data) == 0:
        return px.scatter(title="No regions to analyze"), ""

    diameters = [region.equivalent_diameter for region in regions_data]

    fig = px.histogram(
        x=diameters,
        nbins=40,
        title="Particle Diameter Distribution",
        labels={'x': 'Diameter (pixels)', 'y': 'Count'}
    )

    d10, d50, d90 = np.percentile(diameters, [10, 50, 90])
    fig.add_vline(x=d50, line_dash="dash", line_color="red",
                  annotation_text=f"D50: {d50:.2f}")

    fig.update_layout(
        xaxis_title="Diameter (pixels)",
        yaxis_title="Frequency",
        showlegend=False
    )

    stats = html.Div([
        html.H6("Diameter Statistics:"),
        html.P(f"D10: {d10:.2f} px"),
        html.P(f"D50: {d50:.2f} px"),
        html.P(f"D90: {d90:.2f} px"),
        html.P(f"Mean diameter: {np.mean(diameters):.2f} px"),
        html.P(f"Median diameter: {np.median(diameters):.2f} px")
    ])

    return fig, stats

# Callback for saving analysis results
@app.callback(
    [Output('save-status', 'children'),
     Output('analysis-log-table', 'data'),
     Output('log-summary', 'children')],
    [Input('save-analysis-btn', 'n_clicks'),
     Input('clear-log-btn', 'n_clicks'),
     Input('delete-selected-btn', 'n_clicks')],
    [State('threshold-slider', 'value'),
     State('threshold-2-slider', 'value'),
     State('binary-mode', 'value'),
        State('scale-factor', 'value'),
     State('analysis-log-table', 'selected_rows')]
)
def handle_analysis_log(save_clicks, clear_clicks, delete_clicks, threshold, threshold_2, binary_mode, scale_factor, selected_rows):
    global analysis_log, current_filename, binary_data, regions_data, image_data
    
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    if trigger_id == 'clear-log-btn' and clear_clicks:
        analysis_log = []
        return html.Div("Analysis log cleared!", className="alert alert-info"), [], html.Div()
    
    elif trigger_id == 'delete-selected-btn' and delete_clicks:
        if selected_rows:
            # Remove selected rows in reverse order to maintain indices
            for index in sorted(selected_rows, reverse=True):
                if 0 <= index < len(analysis_log):
                    analysis_log.pop(index)
            
            message = html.Div(f"Deleted {len(selected_rows)} selected row(s)!", 
                             className="alert alert-warning")
            return message, analysis_log, create_log_summary()
        else:
            return html.Div("No rows selected for deletion!", 
                          className="alert alert-warning"), analysis_log, create_log_summary()
    
    elif trigger_id == 'save-analysis-btn' and save_clicks:
        if image_data is None or binary_data is None:
            return html.Div("No analysis to save. Please process an image first!", 
                          className="alert alert-warning"), analysis_log, create_log_summary()

        # Ensure regions_data matches current mode
        if binary_mode == 'particle':
            _, regions_data = anaylyze_particle_regions(binary_data, image_data)
        else:
            _, regions_data = analyze_regions(binary_data, image_data)
        
        # Calculate analysis metrics
        porosity = np.sum(binary_data) / binary_data.size * 100
        
        pore_count = len(regions_data) if regions_data else 0
        mean_pore_size = np.mean([region.area for region in regions_data]) if regions_data else 0
        total_dark_area = sum(region.area for region in regions_data) if regions_data else 0

        diameters = [region.equivalent_diameter for region in regions_data] if regions_data else []
        if diameters:
            d10, d50, d90 = np.percentile(diameters, [10, 50, 90])
        else:
            d10 = d50 = d90 = None

        scale_value = scale_factor if scale_factor and scale_factor > 0 else None
        if scale_value:
            mean_pore_size = mean_pore_size * (scale_value ** 2)
            total_dark_area = total_dark_area * (scale_value ** 2)
            if d10 is not None:
                d10 *= scale_value
                d50 *= scale_value
                d90 *= scale_value
        
        # Create new log entry
        new_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'filename': current_filename or 'Unknown',
            'threshold': threshold,
            'threshold_2': threshold_2 if threshold_2 and threshold_2 > 0 else None,
            'binary_mode': binary_mode,
            'scale_factor': scale_value,
            'd10': d10,
            'd50': d50,
            'd90': d90,
            'porosity': porosity,
            'pore_count': pore_count,
            'mean_pore_size': mean_pore_size,
            'total_dark_area': total_dark_area
        }
        
        analysis_log.append(new_entry)
        
        success_msg = html.Div([
            html.Strong("Analysis saved successfully!"),
            html.Br(),
            html.Small(f"File: {current_filename}, Porosity: {porosity:.2f}%, Pores: {pore_count}")
        ], className="alert alert-success")
        
        return success_msg, analysis_log, create_log_summary()
    
    return html.Div(), analysis_log, create_log_summary()

def create_log_summary():
    """Create summary statistics for the analysis log"""
    if not analysis_log:
        return html.Div("No analyses logged yet.")
    
    df = pd.DataFrame(analysis_log)
    
    return html.Div([
        html.P(f"Total Images Analyzed: {len(analysis_log)}"),
        html.P(f"Average Porosity: {df['porosity'].mean():.2f}% (±{df['porosity'].std():.2f})"),
        html.P(f"Average Pore Count: {df['pore_count'].mean():.1f}"),
        html.P(f"Average Mean Pore Size: {df['mean_pore_size'].mean():.1f} pixels"),
    ], className="bg-light p-3 rounded")

# Callback for CSV export
@app.callback(
    Output("download-csv", "data"),
    Input("export-csv-btn", "n_clicks"),
    prevent_initial_call=True,
)
def export_analysis_log(n_clicks):
    if n_clicks and analysis_log:
        df = pd.DataFrame(analysis_log)
        return dcc.send_data_frame(df.to_csv, "sem_analysis_log.csv", index=False)
    return None

# Callback to handle row selection and delete button state
@app.callback(
    [Output('delete-selected-btn', 'disabled'),
     Output('selection-info', 'children')],
    [Input('analysis-log-table', 'selected_rows')]
)
def update_delete_button(selected_rows):
    if selected_rows:
        count = len(selected_rows)
        button_disabled = False
        info_text = f"{count} row{'s' if count > 1 else ''} selected"
        return button_disabled, info_text
    else:
        button_disabled = True
        info_text = "Select rows to delete"
        return button_disabled, info_text

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port, debug=False)
