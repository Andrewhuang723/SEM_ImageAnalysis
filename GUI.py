import imageio.v2 as imageio
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly
from skimage import morphology
from skimage import measure,color, filters
from skimage.io import imread
from dash import Dash, html, dcc, Input, Output, callback, State
import scipy.ndimage as ndi
import dash_table
from dash_table.Format import Format
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import plotly.express as px
from typing import List


## Read image
img = imread("./SEM-1_i001.tif", as_gray=True)[:1750, :]

## Properties
rows, cols = img.shape[0], img.shape[1]
properties = ['area', 'eccentricity', 'perimeter']


## Parameters
threshold = 100
min_size = 1


def plot_original_figure(img: np.array):
    fig = px.imshow(img, origin="lower", binary_string=True)
    return fig

def img_preprocess(img: np.array, threshold=None, min_size=None) -> np.array:
    ## median denoised
    denoised_img = filters.median(img)

    ## gausssian denoised
    denoised_img = ndi.gaussian_filter(img, sigma=1/4)

    if threshold is None:
        threshold = filters.threshold_otsu(denoised_img)
    if min_size is None:
        min_size = 50
    bw =morphology.closing(denoised_img < threshold)
    bw = morphology.remove_small_objects(bw, min_size=min_size)
    return bw


## First image
# original_fig = plot_original_figure(img)

# ## Second image
# preprocessed_img = img_preprocess(img, threshold=threshold, min_size=min_size)

def get_scaler(txt_path: str):
    
    ## Scaler
    with open(txt_path, "r", encoding="utf-16") as f:
        for line in f.readlines():
            if line[:9] == "PixelSize":
                scaler = float(line[10:])
    
    return scaler
    


## Area regions porporties
def generate_regions_properties(img, scaler=None) -> pd.DataFrame:
    labels = measure.label(img)

    regions_table = measure.regionprops_table(
        label_image=labels, intensity_image=img, properties=properties
    )

    table = pd.DataFrame(regions_table)
    if scaler:
        table['area'] *= scaler
    return table

    


## Third image
def img_with_contour(img: np.array, label_img: np.array, regions: List):
    # plot the original image
    fig = px.imshow(img, origin="lower", binary_string=True)

    for index in range(label_img.max()):
        print("Running contours lines .... (%d/%d)" % (index, label_img.max()))
        label = regions[index].label
        contour = measure.find_contours(label_img == label, level=0.5)[0]
        y, x = contour.T
        hoverinfo = ''
        for prop_name in properties:
            hoverinfo += f'<b>{prop_name}: {getattr(regions[index], prop_name):.2f}</b><br>'
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

    return fig


## Foruth image
## Porosity distriution
def plot_pore_distribution(property_table: pd.DataFrame):
    counts, bins = np.histogram(property_table["area"], bins=100)
    fig = go.Figure(data=[go.Bar(x=bins, y=counts, width=1, marker_color='darkblue')])

    # 设置图表布局
    fig.update_layout(
        title='Pore Size Distribution<br>' + f'Number of pores: {len(property_table)}',
        xaxis=dict(title='Pore Size'),
        yaxis=dict(title='Count'),
    )
    return fig




### return analysis table
# scaler = 24.80469 #um
def generate_analysis_table(property_table, scaler):
    pixels = rows * cols
    real_area = pixels * (scaler ** 2)
    porosity = sum(property_table["area"]) / (scaler * pixels)

    metrics = [
        {"Name": "Total Pixels", "Value": pixels},
        {"Name": "Total Area (um2)", "Value": real_area},
        {"Name": "Porosity (%)", "Value": porosity},
        {"Name": "Pore Area (pixels)", "Value": pixels * porosity},
        {"Name": "Pore Area (um2)", "Value": real_area * porosity},
        {"Name": "Material Area (pixels)", "Value": pixels * (1 - porosity)},
        {"Name": "Matreial Area (um2)", "Value": real_area * (1 - porosity)}
    ]
    return metrics



app = Dash(__name__)
app.layout = html.Div(className="container",
                    children=[
                        html.Div([
                            html.H1("Foxconn Battery Cell Analysis Lab"),
                            html.H2("Pore Analysis for SEM image")
                        ]),
                        ## Upload image
                        html.Div([
                            html.Div([
                                html.H3("Upload SEM image"),
                                dcc.Upload(
                                    id='upload-image',
                                    children=html.Div([
                                        'Drag and Drop or ',
                                        html.A('Select Files')
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
                                    # Allow multiple files to be uploaded
                                    multiple=False
                                ),
                                html.Div(id="original_image"),
                                
                                html.H3("Upload SEM info"),
                                dcc.Upload(
                                    id='upload-info',
                                    children=html.Div([
                                        'Drag and Drop or ',
                                        html.A('Select Files')
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
                                    # Allow multiple files to be uploaded
                                    multiple=False
                                ),
                                html.Div(id='original_info'),
                            ])
                        ]),
                        ## Show original image
                        html.Div([
                            html.H3("Original Image"),
                            dcc.Graph(
                                id="original-image-plot",
                                style={'width': '80', 'height': '50vh'}
                            )
                        ]),
                        html.Div([
                            html.H3("Threshold"),
                            dcc.Input(
                                id='threshold-input',
                                type='number',
                                value=90, min=0, max=255
                            ),
                            dcc.Graph(
                                id="preprocessed_image",
                                style={'width': '80', 'height': '50vh'}
                            ),
                            html.Table(id="property-table")
                        ]),
                        html.Div([
                            html.H3("Pore Distribution"),
                            dcc.Graph(
                                id="pore-distribution",
                                style={'width': '80', 'height': '50vh'}
                            )
                        ]),

                        html.Div([
                            html.H3("Pore Visualization"),
                                html.Button(
                                    children='RUN', 
                                    id='run-pore-visualization', 
                                    n_clicks=0,
                                    style={'background-color': 'blue', 'color': 'white'}),
                            dcc.Graph(
                                id="contour_image",
                                style={'width': '80', 'height': '50vh'}
                            )
                        ]),

                        html.Div([
                            html.H4('Results table'),
                            html.P(id='table_out'),
                        ])
                    ]
)


import datetime
import io
import base64
from PIL import Image

def parse_contents_to_array(contents):
    _, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    image = Image.open(io.BytesIO(decoded))
    return np.array(image)

def parse_contents_to_string(contents):
    _, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    text = decoded.decode('utf-16')
    return text



@callback(Output('original_image', 'children'),
          Output('original-image-plot', 'figure'),
          Input("upload-image", "contents"))
def update_image(contents):
    img_arr = parse_contents_to_array(contents)
    fig = plot_original_figure(img_arr)
    return contents, fig



@callback(
        Output("original_info", 'children'),
        Input("upload-info", "contents"),
)
def update_info(contents):
    text = parse_contents_to_string(contents)
    for line in text.rsplit():
        if line[:9] == "PixelSize":
            scaler = float(line[10:])
    return scaler

@callback(
        Output("threshold-input", "value"),
        Input("threshold-input", "value")
)
def threshold_defined(threshold):
    return threshold

@callback(
        Output("preprocessed_image", "fig"),
        Output("property-table", "table"),
        Input("original_image", "children"),
        Input("threshold-input", "value"),
        State("original_info", "children")
)
def preprocessed_image(contents, threshold, info):
    img_arr = parse_contents_to_array(contents)
    bw = img_preprocess(img_arr, threshold=threshold, min_size=min_size)
    fig = px.imshow(bw, binary_string=True, title="threshold: %d" % threshold)

    table = generate_regions_properties(img=bw, scaler=info)
    return fig, table


@callback(
        Output("pore-distribution", "children"),
        Input("property_table", "table")
)
def distribution(table: pd.DataFrame):
    fig = plot_pore_distribution(property_table=table)
    return fig



@callback(
        Output("contour_image", "fig"),
        State("original_image", "contents"),
        State("preprocessed_image", "contents"),
        Input("run-pore-visualization", "n_clicks")

)
def visualized_pore(n_clicks, contents, bw):
    img_arr = parse_contents_to_array(contents)
    label_img = measure.label(bw, connectivity=None)
    regions = measure.regionprops(label_img)
    return img_with_contour(img_arr, label_img=label_img, regions=regions)


@callback(
    Output('table_out', 'children'),
    Input('preprocessed_image', 'children'),
    State('upload-info', 'children'))
def update_graphs(bw, scaler):
    regions_table = generate_regions_properties(bw, scaler=scaler)
    metrics = generate_analysis_table(property_table=regions_table, scaler=scaler)
    
    return dash_table.DataTable(
                                columns=[{"name": "Name", "id":"Name"}, {"name":"Value", "id":"Value"}],
                                data=metrics,
                                style_cell=dict(textAlign='left', color='white'),
                                style_header=dict(backgroundColor="darkblue", fontWeight="bold", fontColor="white"),
                                style_data=dict(backgroundColor="black")
                            )


app.run_server(debug=True)