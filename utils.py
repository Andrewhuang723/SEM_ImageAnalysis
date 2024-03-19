import dash
import dash_table
from dash_table.Format import Format
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import datetime
import io
import base64
from PIL import Image
import numpy as np
from skimage import io, filters, measure, color, img_as_ubyte
import PIL
import pandas as pd
import matplotlib as mpl

external_stylesheets = [dbc.themes.BOOTSTRAP, "assets/object_properties_style.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

filename = "./image/Chang LLZAO.tif"

prop_names = [
        "label",
        "area",
        "perimeter",
        "eccentricity",
        "euler_number",
        "mean_intensity",
    ]


img_upload = dbc.Card(
    [dcc.Upload(
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
    multiple=False,
    ),
    html.Div(id="output-img")]
)

threhold_input = dbc.Card(
    [dcc.Input(
        id='threshold-input',
        type='number',
        value=90, min=0, max=255
    )]
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
        label = row.label
        contour = measure.find_contours(label_img == label, level=0.5)[0]
        y, x = contour.T
        hoverinfo = ''
        for prop_name in prop_names:
            hoverinfo += f'<b>{prop_name}: {getattr(rid, prop_name):.2f}</b><br>'
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
                    dbc.Col(dbc.NavbarBrand("Object Properties App")),
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



# Color selector dropdown
# color_drop = dcc.Dropdown(
#     id="color-drop-menu",
#     options=[
#         {"label": col_name.capitalize(), "value": col_name}
#         for col_name in table.columns
#     ],
#     value="label",
# )

# Define Cards

image_card = lambda fig: dbc.Card(
    [
        dbc.CardHeader(html.H2("Explore object properties")),
        dbc.CardBody(
            dbc.Row(
                dbc.Col(
                    dcc.Graph(
                        id="graph",
                        figure=fig,
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
# table_card = dbc.Card(
#     [
#         dbc.CardHeader(html.H2("Data Table")),
#         dbc.CardBody(
#             dbc.Row(
#                 dbc.Col(
#                     [
#                         dash_table.DataTable(
#                             id="table-line",
#                             columns=columns,
#                             data=table.to_dict("records"),
#                             tooltip_header={
#                                 col: "Select columns with the checkbox to include them in the hover info of the image."
#                                 for col in table.columns
#                             },
#                             style_header={
#                                 "textDecoration": "underline",
#                                 "textDecorationStyle": "dotted",
#                             },
#                             tooltip_delay=0,
#                             tooltip_duration=None,
#                             filter_action="native",
#                             row_deletable=True,
#                             column_selectable="multi",
#                             selected_columns=initial_columns,
#                             style_table={"overflowY": "scroll"},
#                             fixed_rows={"headers": False, "data": 0},
#                             style_cell={"width": "85px"},
#                         ),
#                         html.Div(id="row", hidden=True, children=None),
#                     ]
#                 )
#             )
#         ),
#     ]
# )


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

# regions_table = generate_regions_properties(bw, scaler=scaler)
# metrics = generate_analysis_table(property_table=table, scaler=317)

metric_card = lambda property_table: dbc.Card(
    [
        dbc.CardHeader(html.H2("Results Table")),
        dbc.CardBody(
            dbc.Row(
                dbc.Col(
                    [
                        dash_table.DataTable(
                                columns=[{"name": "Name", "id":"Name"}, {"name":"Value", "id":"Value"}],
                                data=generate_analysis_table(property_table=property_table),
                                style_cell=dict(textAlign='left', color='white'),
                                style_header=dict(backgroundColor="darkblue", fontWeight="bold", fontColor="white"),
                                style_data=dict(backgroundColor="black")
                        ),
                        html.Div(id="results", hidden=True, children=None),
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
            [   dbc.Row(dbc.Col(img_upload, md=2), dbc.Col(threhold_input, md=2)),
                dbc.Row([dbc.Col(image_card, md=5), dbc.Col(metric_card, md=5)])],
            fluid=True,
        ),
    ]
)






@app.callback(
        Output('output-img', 'children'),
        Input("upload-image", "contents")
)
def update_image(contents):
    return contents


@app.callback(
        Output("threshold-input", "value"),
        Input("threshold-input", "value")
)
def threshold_defined(threshold):
    return threshold

@app.callback(
        Output("graph", "fig"),
        Output("results", "children"),
        Input("output-img", "contents"),
        State("threshold-input", "value")
)
def contour(contents, threshold):
    img = parse_contents_to_array(contents=contents)
    table, prep_img = get_preprocessed_img(img=img, threshold=threshold)
    contour_img = img_with_contour(img=img, label_img=prep_img, region_table=table)
    metrics_table = generate_analysis_table(property_table=table)
    return contour_img, metrics_table




if __name__ == "__main__":
    app.run_server(debug=False)