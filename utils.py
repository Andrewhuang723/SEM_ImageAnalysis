import dash
import dash_table
from dash_table.Format import Format
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.offline as py
import datetime
import io
from io import StringIO
import base64
from PIL import Image
import numpy as np
from skimage import filters, measure, color, img_as_ubyte
import PIL
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time
from functools import lru_cache


external_stylesheets = [dbc.themes.BOOTSTRAP, "assets/object_properties_style.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server


prop_names = [
        "label",
        "area",
        "perimeter",
        "eccentricity",
        "euler_number",
        "mean_intensity",
    ]

@lru_cache(maxsize=32)
def parse_contents_to_array(contents):
    if contents is None:
        return None
        
    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    # Open and optimize image
    image = Image.open(io.BytesIO(decoded))
    image = optimize_image(image)
    
    # Convert to grayscale immediately to reduce memory usage
    if image.mode != 'L':
        image = image.convert('L')
    
    return np.array(image)

def optimize_image(image):
    """Optimize image size and quality before processing"""
    # Convert to RGB if image has alpha channel
    if image.mode in ('RGBA', 'LA'):
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        image = background
    
    # Resize if image is too large
    max_size = 800  # Reduced from 1000 to 800 for faster processing
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    return image

@lru_cache(maxsize=32)
def array_to_base64_str(image_array_tuple):
    # Convert tuple back to numpy array
    image_array = np.array(image_array_tuple)
    
    # Convert the NumPy array to a PIL Image object
    image = Image.fromarray(image_array.astype(np.uint8))
    
    # Save the image object to an in-memory bytes buffer with compression
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85, optimize=True)  # Changed to JPEG for faster encoding
    buffer.seek(0)
    
    # Encode the image bytes in base64
    base64_str = base64.b64encode(buffer.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_str}"

@lru_cache(maxsize=32)
def get_preprocessed_img(img_tuple, threshold=None):
    # Convert tuple back to array since numpy arrays aren't hashable
    img = np.array(img_tuple)
    
    if threshold is None:
        threshold = filters.threshold_otsu(img)

    label_array = measure.label(img < threshold)
    prop_table = measure.regionprops_table(
        label_array, intensity_image=img, properties=prop_names
    )
    table = pd.DataFrame(prop_table)
    
    return table, label_array

def numpy_to_tuple(arr):
    """Convert numpy array to hashable tuple"""
    return tuple(map(tuple, arr.tolist()))

## Thread with img_with_contour
progress = {'value': 0}


def img_with_contour(img: np.array, label_img: np.array, region_table: pd.DataFrame):
    # plot the original image
    fig = px.imshow(img, binary_string=True)

    for rid, row in region_table.iterrows():
        
        progress['value'] = int((rid+1) / label_img.max() * 100)

        label = row.label
        contour = measure.find_contours(label_img == label, level=0.5, fully_connected="high")[0]
        y, x = contour.T
        hoverinfo = ''
        for prop_name in prop_names:
            hoverinfo += f'<b>{prop_name}: {row[prop_name]:.2f}</b><br>'
        fig.add_trace(
            go.Scatter(
                x=x, y=y,
                name=label,
                mode='lines', 
                fill='toself', 
                showlegend=False,
                hovertemplate=hoverinfo, 
                hoveron='points+fills'
            )
        )
    fig.update_layout(
            title=f'Number of pores: {len(region_table)}'
        )


    return fig


def plot_pore_distribution(property_table: pd.DataFrame, scaler=1):
    
    property_table["rescale_area"] = property_table["area"].apply(lambda x: x * (scaler ** 2))
    fig = px.histogram(property_table, x='rescale_area', marginal='rug', color_discrete_sequence=['darkblue'])



    fig.update_layout(
        title='Pore Size Distribution<br>' + f'Number of pores: {len(property_table)}',
        xaxis=dict(title=r"$Pore Size(\mu m^2)$"),
        yaxis=dict(title='Count'),
    )
    
    return fig


def plot_grayscale_distribution(img, threshold):
    fig = px.histogram(img.reshape(-1), marginal='rug', color_discrete_sequence=['darkblue'])

    fig.update_layout(
        title='Grayscale Distribution<br>',
        xaxis=dict(title="Gray scale"),
        yaxis=dict(title='Count'),
    )

    fig.add_shape(
    # Line Vertical
    type="line",
    x0=threshold,  # x-coordinate where the line starts
    y0=0,  # y-coordinate where the line starts
    x1=threshold,  # x-coordinate where the line ends
    y1=1,  # y-coordinate where the line ends, 1 means it stretches to the top
    line=dict(
        color="Red",
        width=2
    ),
    xref="x",  # it refers to the x-values of the histogram
    yref="paper"  # it refers to the relative height of the plot
    )

    fig.add_annotation(
        x=threshold,
        text=f"x = {threshold}",  # Text to display
        showarrow=True,
        arrowhead=1,
        arrowsize=1,
        arrowwidth=1,
        arrowcolor="Red",
        ax=20  # Adjust this to move the text if needed
    )
    
    return fig

def generate_analysis_table(img, property_table, scaler):
    rows, cols = img.shape[0], img.shape[1]
    pixels = rows * cols
    real_area = pixels * (scaler ** 2)
    porosity = sum(property_table["area"]) / (pixels)

    metrics = [
        {"Name": "Total Pixels", "Value": pixels},
        {"Name": "Total Area (um2)", "Value": real_area},
        {"Name": "Porosity", "Value": porosity},
        {"Name": "Pore Area (pixels)", "Value": pixels * porosity},
        {"Name": "Pore Area (um2)", "Value": real_area * porosity},
        {"Name": "Material Area (pixels)", "Value": pixels * (1 - porosity)},
        {"Name": "Matreial Area (um2)", "Value": real_area * (1 - porosity)}
    ]
    return metrics

def generate_describe_table(img, property_table, scaler):
    property_table["area"] *= (scaler ** 2)
    property_table["perimeter"] *= scaler    
    describe_table = property_table.describe()
    
    metrics = [
        {"Name": 'mean', "Area": describe_table.loc["mean", "area"], "Perimeter": describe_table.loc["mean", "perimeter"], "Intensity": describe_table.loc["mean", "mean_intensity"]},
        {"Name": 'std', "Area": describe_table.loc["std", "area"], "Perimeter": describe_table.loc["std", "perimeter"], "Intensity": describe_table.loc["std", "mean_intensity"]},
        {"Name": 'min', "Area": describe_table.loc["min", "area"], "Perimeter": describe_table.loc["min", "perimeter"], "Intensity": describe_table.loc["min", "mean_intensity"]},
        {"Name": '25%', "Area": describe_table.loc["25%", "area"], "Perimeter": describe_table.loc["25%", "perimeter"], "Intensity": describe_table.loc["25%", "mean_intensity"]},
        {"Name": '50%', "Area": describe_table.loc["50%", "area"], "Perimeter": describe_table.loc["50%", "perimeter"], "Intensity": describe_table.loc["50%", "mean_intensity"]},
        {"Name": '75%', "Area": describe_table.loc["75%", "area"], "Perimeter": describe_table.loc["75%", "perimeter"], "Intensity": describe_table.loc["75%", "mean_intensity"]},
        {"Name": 'max', "Area": describe_table.loc["max", "area"], "Perimeter": describe_table.loc["max", "perimeter"], "Intensity": describe_table.loc["max", "mean_intensity"]}
    ]
    return metrics


with open("README.md", "r") as f:
    howto_md = f.read()


modal_overlay = dbc.Modal(
    [
        dbc.ModalBody(html.Div([dcc.Markdown(howto_md)], id="howto-md")),
        dbc.ModalFooter(dbc.Button("Close", id="howto-close", className="howto-bn")),
    ],
    id="modal",
    size="lg",
)

# Buttons
button_gh = dbc.Button(
    "Learn more",
    id="howto-open",
    outline=True,
    color="secondary",
    # Turn off lowercase transformation for class .button in stylesheet
    style={"textTransform": "none"},
)

button_howto = dbc.Button(
    "View Code on github",
    outline=True,
    color="primary",
    href="https://github.com/Andrewhuang723/SEM_ImageAnalysis",
    id="gh-link",
    style={"text-transform": "none"},
)

# Define Header Layout
header = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.A(
                            html.Img(
                                src="assets/Foxconn.png",
                                height="30px",
                            ),
                            href="https://www.foxconn.com/",
                        )
                    ),
                    dbc.Col(dbc.NavbarBrand("A SEM Image Pore Analysis App by 高階分析實驗室")),
                    modal_overlay,
                ],
                align="center",
            ),
            dbc.Row(
                dbc.Col(
                    [
                        dbc.NavbarToggler(id="navbar-toggler"),
                        dbc.Collapse(
                            dbc.Nav(
                                [dbc.NavItem(button_howto), dbc.NavItem(button_gh)],
                                className="ml-auto",
                                navbar=True,
                            ),
                            id="navbar-collapse",
                            navbar=True,
                        ),
                    ]
                ),
                align="center",
            ),
        ],
        fluid=True,
    ),
    color="dark",
    dark=True,
)

img_upload = dbc.Card(
    dbc.CardBody([
    dcc.Upload(
    id='upload-img',
    children=html.Div([
        'Drag and Drop or ',
        html.A('Select Files')
    ]),
    accept='image/*',  # Only accept image files
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
    # Allow multiple files to be uploaded
    multiple=False,
    ),
    html.Div([
        dbc.Progress(id="image-progress", value=0, label="0%", striped=True, animated=True),
    ], style={'margin': '10px'}),
    dbc.Row([
        dbc.Col(
            html.Div([
                html.H3("Original Image", style={'color': 'black'}),
                html.Img(id='output-img-str', src='', style={'max-width': '70%', 'max-height': '80%',}),
            ]),
        ), 
        dbc.Col(
                html.Div([
                html.H3("Filtered Image", style={'color': 'black'}),
                html.Img(id='output-preprocessed-img-str', src='', style={'max-width': '70%', 'max-height': '80%',}),
            ])
        )
    ])
    
])
)

threshold_input = dbc.Card(
    dbc.CardBody([
        html.H4("Threshold", className="card-title"),
        dbc.Row([
            dbc.Col(
                dbc.Input(type="number", value=90, min=0, max=255, id='threshold-input'),
                width=8
            ),
            dbc.Col(
                dbc.Button("Apply", id="apply-threshold", color="primary"),
                width=4
            )
        ]),
        html.P("* threshold = grayscale", className="card-text")
    ])
)

scaler_input = dbc.Card(
    dbc.CardBody([
        html.H4("Scaler", className="card-title"),
        dbc.Input(type="number", value=1, id='scaler-input'),
        html.P(r"$* 1 pixel = __ um$", className="card-text")
    ])
)

# Define Cards
image_card = dbc.Card(
    [
        dbc.CardHeader(html.H2("Pores location", style={'color': 'black'})),
        html.Div([
                    dbc.Button(
                    'RUN', 
                    id='run-contour-plot', 
                    n_clicks=0,
                    style={'background-color': 'blue', 'color': 'white'})
                    ]),
        html.Div([
                dcc.Interval(id="progress-interval", interval=500),
                dbc.Progress(id="progress-bar", value=0, label="0%", striped=True, animated=True, style={'width': '50%'}),
            ]),
        dbc.CardBody(              
            dbc.Row(
                dbc.Col(
                    dcc.Graph(
                        id="graph",
                        style={'width': '200', 'height': '60vh'}
                    ),
                )
            )
        ),
        dbc.CardFooter(
            dbc.Row(
                [
                    dbc.Col(
                        ["Use the dropdown menu to select which variable to base the colorscale on:",
                        html.Button("Download CSV", id="contour-download"),
                        dcc.Download(id="download-contour-csv"),]
                    )
                ],
                align="center",
            )),
])






distribution_card = dbc.Card([
    dbc.CardHeader(html.H2("Pore Distribution", style={'color': 'black'})),
    dbc.CardBody([
        html.H4("Distribution plot", className="card-title"),
        dcc.Graph(id='distribution-plot'),
        html.P("Adjust the threshold.", className="card-text")
    ])
])

gray_scale_distribution_card = dbc.Card([
    dbc.CardHeader(html.H2("Gray scale Distribution", style={'color': 'black'})),
    dbc.CardBody([
        html.H4("Gray Scale Distribution plot", className="card-title"),
        dcc.Graph(id='gray-scale-distribution-plot'),
        html.P("Adjust the threshold.", className="card-text")
    ])
])

metric_card = dbc.Card(
    [
        dbc.CardHeader(html.H2("Analysis Table", style={'color': 'black'})),
        dbc.CardBody(
            dbc.Row(
                dbc.Col(
                    [
                        dash_table.DataTable(
                                id = "results",
                                columns=[{"name": "Name", "id":"Name"}, {"name":"Value", "id":"Value"}],
                                style_cell=dict(textAlign='left', color='white'),
                                style_header=dict(backgroundColor="darkblue", fontWeight="bold", fontColor="white"),
                                style_data=dict(backgroundColor="black")
                        ),
                        html.Button("Download CSV", id="table-download"),
                        dcc.Download(id="download-table-csv"),
                    ]
                )
            ),
        ),
    ]
)


describe_card = dbc.Card(
    [
        dbc.CardHeader(html.H2("Describe Table", style={'color': 'black'})),
        dbc.CardBody(
            dbc.Row(
                dbc.Col(
                    [
                        dash_table.DataTable(
                                id = "describe",
                                columns=[{"name": "Name", "id":"Name"}, {"name":"Area", "id":"Area"}, {"name":"Perimeter", "id":"Perimeter"}, {"name":"Intensity", "id":"Intensity"}],
                                style_cell=dict(textAlign='left', color='white'),
                                style_header=dict(backgroundColor="darkblue", fontWeight="bold", fontColor="white"),
                                style_data=dict(backgroundColor="black")
                        ),
                        html.Button("Download CSV", id="describe-download"),
                        dcc.Download(id="download-describe-csv"),
                    ]
                )
            ),
        ),
    ]
)


app.layout = html.Div(
    [
        header,        
        dbc.Container(
            [dbc.Row([dbc.Col(img_upload, md=10),  
                      dbc.Col([dbc.Row(threshold_input), 
                              dbc.Row(scaler_input)], md=2)]),
             dbc.Row([dbc.Col(gray_scale_distribution_card, md=10)]),
             dbc.Row([dbc.Col(image_card, md=10)]),
             dbc.Row([dbc.Col(distribution_card, md=10)]),
             dbc.Row([dbc.Col(metric_card, md=8)]),
             dbc.Row([dbc.Col(describe_card, md=8)])
             ],
            
            fluid=True,
        ),
    ]
)

@app.callback(
        Output("output-img-str", "src"),
        Output("output-preprocessed-img-str", "src"),
        Output("gray-scale-distribution-plot", "figure"),
        Output("image-progress", "value"),
        Output("image-progress", "label"),
        Input("upload-img", "contents"),
        Input("apply-threshold", "n_clicks"),
        State("threshold-input", "value"),
        State("scaler-input", "value")
)
def update_image(contents, n_clicks, threshold, scaler):
    if contents is None:
        raise dash.exceptions.PreventUpdate
        
    # Initialize progress
    progress = 0
    
    try:
        # Parse and optimize image (25%)
        img = parse_contents_to_array(contents)
        if img is None:
            raise ValueError("Failed to parse image")
        progress = 25
        
        # Process image and convert to base64 in parallel (75%)
        with ThreadPoolExecutor(max_workers=2) as executor:
            img_tuple = numpy_to_tuple(img)
            future_base64 = executor.submit(array_to_base64_str, img_tuple)
            future_processed = executor.submit(get_preprocessed_img, img_tuple, threshold)
            
            array_str = future_base64.result()
            table, prep_img = future_processed.result()
            progress = 75
        
        # Final conversion (100%)
        prep_img_tuple = numpy_to_tuple(prep_img)
        prep_arr_str = array_to_base64_str(prep_img_tuple)
        progress = 100

        ## gray-scale distribution
        fig = plot_grayscale_distribution(img=img, threshold=threshold)
        
        return array_str, prep_arr_str, fig, progress, f"{progress}%"
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise dash.exceptions.PreventUpdate


@app.callback(
    Output('progress-bar', 'value'),
    Output('progress-bar', 'label'),
    Input('progress-interval', 'n_intervals'),
    prevent_initial_call=True
)
def update_progress(n):
    return progress['value'], f"{progress['value']}%"



executor = ThreadPoolExecutor(max_workers=1) 


@app.callback(
    Output('run-contour-plot', 'n_clicks'),
    Output('graph', 'figure'),
    Output("distribution-plot", "figure"),
    Output("results", "data"),
    Output("describe", "data"),
    Input('run-contour-plot', 'n_clicks'),
    State("output-img-str", "src"),
    State("threshold-input", "value"),
    State("scaler-input", "value"),
    prevent_initial_call=True
)
def start_long_process(n_clicks, src, threshold, scaler):
    contents = src
    img = parse_contents_to_array(contents=contents)
    img_tuple = numpy_to_tuple(img)
    table, prep_img = get_preprocessed_img(img_tuple, threshold)

    if n_clicks > 0:
        future = executor.submit(img_with_contour, img, prep_img, table)
        contour = future.result()
    distribution_plot = plot_pore_distribution(property_table=table, scaler=scaler)
    metrics_table = generate_analysis_table(img=img, property_table=table, scaler=scaler)
    describe_table = generate_describe_table(img=img, property_table=table, scaler=scaler)
    return True, contour, distribution_plot, metrics_table, describe_table


@app.callback(
    Output("download-contour-csv", "data"),
    Input("contour-download", "n_clicks"),
    State("output-img-str", "src"),
    State("threshold-input", "value"),
    State("scaler-input", "value"),
    prevent_initial_call=True
)
def download_pores_csv(n_clicks, src, threshold, scaler):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    contents = src
    img = parse_contents_to_array(contents=contents)

    # Convert DataFrame to a CSV string buffer
    buffer = StringIO()
    df, _ = get_preprocessed_img(numpy_to_tuple(img), threshold)
    df["area"] *= (scaler ** 2)
    df["perimeter"] *= scaler

    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return dcc.send_string(buffer.getvalue(), "pores.csv")


@app.callback(
    Output("download-table-csv", "data"),
    Input("table-download", "n_clicks"),
    State("results", "data"),
    prevent_initial_call=True
)
def download_metrics_csv(n_clicks, table):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    # Convert DataFrame to a CSV string buffer
    buffer = StringIO()
    df = pd.DataFrame(table)
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return dcc.send_string(buffer.getvalue(), "table.csv")


@app.callback(
    Output("download-describe-csv", "data"),
    Input("describe-download", "n_clicks"),
    State("describe", "data"),
    prevent_initial_call=True
)
def download_metrics_csv(n_clicks, table):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    # Convert DataFrame to a CSV string buffer
    buffer = StringIO()
    df = pd.DataFrame(table)
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return dcc.send_string(buffer.getvalue(), "table.csv")


if __name__ == "__main__":
    app.run_server(debug=False, port=8000)