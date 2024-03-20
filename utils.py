import dash
import dash_table
from dash_table.Format import Format
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import datetime
import io
import base64
from PIL import Image
import numpy as np
from skimage import filters, measure, color, img_as_ubyte
import PIL
import pandas as pd
import matplotlib as mpl

external_stylesheets = [dbc.themes.BOOTSTRAP, "assets/object_properties_style.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# filename = "./image/Chang LLZAO.tif"

prop_names = [
        "label",
        "area",
        "perimeter",
        "eccentricity",
        "euler_number",
        "mean_intensity",
    ]


img_upload = dbc.Card(
    dbc.CardBody([
    dcc.Upload(
    id='upload-img',
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
    multiple=False,
    ),
    html.Div([
        html.H3("Original Image"),
        dcc.Graph(
            id="output-img",
            style={'width': '80', 'height': '50vh'}
        ),
        
    ]),
])
)

threshold_input = dbc.Card(
    dbc.CardBody([
        html.H4("Threshold", className="card-title"),
        dbc.Input(type="number", value=90, min=0, max=255, id='threshold-input'),
        html.P("* threshold = grayscale", className="card-text")
    ])
)


def parse_contents_to_array(contents):
    _, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    image = Image.open(io.BytesIO(decoded))
    return np.array(image)


def get_preprocessed_img(img, threshold=None):
    # img = io.imread(filename, as_gray=True)
    rows, cols = img.shape[0], img.shape[1]
    
    if threshold is None:
        threshold = filters.threshold_otsu(img)

    label_array = measure.label(img < threshold)
    # Compute and store properties of the labeled image
    
    prop_table = measure.regionprops_table(
        label_array, intensity_image=img, properties=prop_names
    )
    table = pd.DataFrame(prop_table)
    
    return table, label_array


def img_with_contour(img: np.array, label_img: np.array, region_table: pd.DataFrame):
    # plot the original image
    fig = px.imshow(img, origin="lower", binary_string=True)

    for rid, row in region_table.iterrows():
        print("Running contours lines .... (%d/%d)" % (rid, label_img.max()))
        fig.update_layout(
            title="Running contours lines .... "
        )
        label = row.label
        contour = measure.find_contours(label_img == label, level=0.5)[0]
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
    href="https://github.com/plotly/dash-sample-apps/tree/master/apps/dash-label-properties",
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
                                src=app.get_asset_url("dash-logo-new.png"),
                                height="30px",
                            ),
                            href="https://plotly.com/dash/",
                        )
                    ),
                    dbc.Col(dbc.NavbarBrand("SEM Image Pore Analysis App")),
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

# Define Cards

image_card = dbc.Card(
    [
        dbc.CardHeader(html.H2("Explore object properties")),
        html.Div([
                    dbc.Button(
                    'RUN', 
                    id='run-contour-plot', 
                    n_clicks=0,
                    style={'background-color': 'blue', 'color': 'white'})
                    ]), 
        dbc.CardBody(  
            dbc.Row(
                dbc.Col(
                    dcc.Graph(
                        id="graph",
                    ),
                )
            )
        ),
        dbc.CardFooter(
            dbc.Row(
                [
                    dbc.Col(
                        "Use the dropdown menu to select which variable to base the colorscale on:"
                    ),
                    # dbc.Col(color_drop),
                    dbc.Toast(
                        [
                            html.P(
                                "In order to use all functions of this app, please select a variable "
                                "to compute the colorscale on.",
                                className="mb-0",
                            )
                        ],
                        id="auto-toast",
                        header="No colorscale value selected",
                        icon="danger",
                        style={
                            "position": "fixed",
                            "top": 66,
                            "left": 10,
                            "width": 350,
                        },
                    ),
                ],
                align="center",
            ),
        ),
    ]
)


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



distribution_card = dbc.Card(
    dbc.CardBody([
        html.H4("Distribution plot", className="card-title"),
        dcc.Graph(id='distribution-plot'),
        html.P("Adjust the threshold.", className="card-text")
    ])
)

metric_card = dbc.Card(
    [
        dbc.CardHeader(html.H2("Results Table")),
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
                        # html.Div(id="results", hidden=True, children=None),xs
                    ]
                )
            )
        ),
    ]
)

app.layout = html.Div(
    [
        header,        
        dbc.Container(
            [dbc.Row([dbc.Col(img_upload, md=10), dbc.Col(threshold_input, md=2)]),
             dbc.Row([dbc.Col(image_card, md=10)]),
             dbc.Row([dbc.Col(distribution_card, md=10)]),
             dbc.Row([dbc.Col(metric_card, md=8)])
             ],
            
                # dbc.Row([dbc.Col(image_card, md=5), dbc.Col(metric_card, md=5)])],
            fluid=True,
        ),
    ]
)


@app.callback(
        Output('output-img', 'figure'),
        Output("graph", "figure"),
        Output("distribution-plot", "figure"),
        Output("results", "data"),
        Input("upload-img", "contents"),
        Input("threshold-input", "value")
)
def update_image(contents, threshold):
    img = parse_contents_to_array(contents)
    table, prep_img = get_preprocessed_img(img=img, threshold=threshold)
    fig = px.imshow(prep_img, origin="lower", binary_string=True)
    fig.update_layout(
        title=f'Pore definition is the gray scale <= {threshold}',
    )

    contour_img = img_with_contour(img=img, label_img=prep_img, region_table=table)
    distribution_plot = plot_pore_distribution(property_table=table)
    metrics_table = generate_analysis_table(img=img, property_table=table, scaler=317)
    return fig, contour_img, distribution_plot, metrics_table


# @app.callback(
#         Output("threshold-input", "value"),
#         Input("threshold-input", "value")
# )
# def threshold_defined(threshold):
#     return threshold

# @app.callback(
#         Output("graph", "figure"),
#         Output("distribution-plot", "figure"),
#         State("upload-img", "contents"),
#         State("threshold-input", "value"),
#         Input("run-contour-plot", "n_clicks"),
# )
# def contour(n_clicks, threshold, contents):
#     img = parse_contents_to_array(contents=contents)
#     table, prep_img = get_preprocessed_img(img=img, threshold=threshold)
#     contour_img = img_with_contour(img=img, label_img=prep_img, region_table=table)
#     distribution_plot = plot_pore_distribution(property_table=table)
#     # metrics_table = generate_analysis_table(property_table=table)
#     return contour_img, distribution_plot




if __name__ == "__main__":
    app.run_server(debug=False)