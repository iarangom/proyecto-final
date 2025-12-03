import dash
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import plotly.express as px

# Importar funciones de los modelos
from funciones_modelos import predecir_precio, predecir_recomendable

# ==========================
# CARGA DE DATOS Y GRÁFICOS
# ==========================

df = pd.read_csv("airbnb_rio_clean.csv")

# Precio promedio por número de cuartos
if {"price", "bedrooms"}.issubset(df.columns):
    fig_precio_vs_cuartos = px.histogram(
        df,
        x="bedrooms",
        y="price",
        histfunc="avg",
        nbins=20,
        labels={"bedrooms": "Número de cuartos", "price": "Precio promedio por noche"},
        title="Precio promedio por número de cuartos",
    )
else:
    fig_precio_vs_cuartos = px.scatter(
        title="No se encontraron las columnas 'price' y/o 'bedrooms'"
    )

# Distribución de precios por noche
if "price" in df.columns:
    fig_dist_precios = px.histogram(
        df,
        x="price",
        nbins=1000,
        labels={"price": "Precio por noche", "count": "Frecuencia"},
        title="Distribución de precios por noche",
    )

    # Limitar rango del eje X a 0–4000
    fig_dist_precios.update_xaxes(range=[0, 4000])

else:
    fig_dist_precios = px.scatter(title="No se encontró la columna 'price'")


# Host response rate vs recomendación (si existen columnas)
# --- Cantidad de inmuebles y precio promedio por rating ---
# --- Cantidad de inmuebles y precio promedio por rango de rating ---
if {"review_scores_rating", "price"}.issubset(df.columns):

    # crear bins
    bins = np.arange(0, 5.5, 0.5)
    labels = [f"{round(bins[i],1)} - {round(bins[i+1],1)}" for i in range(len(bins)-1)]
    
    df["rating_bin"] = pd.cut(
        df["review_scores_rating"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    # agrupar
    df_group = df.groupby("rating_bin").agg(
        count=("price", "size"),
        avg_price=("price", "mean")
    ).reset_index()

    # eliminar bins vacíos
    df_group = df_group.dropna()

    # gráfica de barras
    fig_resp_vs_reco = px.bar(
        df_group,
        x="rating_bin",
        y="count",
        labels={
            "rating_bin": "Rango de rating",
            "count": "Cantidad de inmuebles"
        },
        title="Cantidad de inmuebles y precio promedio por rango de rating",
        text="avg_price"
    )

    # formatear texto encima de barras
    fig_resp_vs_reco.update_traces(
        texttemplate="$%{text:.0f}", 
        textposition="outside"
    )

    # actualizar layout
    fig_resp_vs_reco.update_layout(
        yaxis=dict(title="Cantidad"),
        xaxis=dict(title="Rango de rating"),
        uniformtext_minsize=8,
        uniformtext_mode="hide",
        bargap=0.1,
        showlegend=False
    )

else:
    fig_resp_vs_reco = px.scatter(
        title="No se encontraron columnas 'review_scores_rating' y/o 'price'"
    )



# ==========================
# CREAR APP
# ==========================

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # por si lo necesitas luego

# ==========================
# LAYOUT
# ==========================

app.layout = html.Div(
    [
        html.H1(
            "Tablero Analítico de Airbnb",
            style={"textAlign": "center", "marginTop": "20px"},
        ),
        dcc.Tabs(
            id="tabs",
            value="tab-general",
            children=[
                dcc.Tab(label="Visión general", value="tab-general"),
                dcc.Tab(
                    label="Predicción de Precio y Recomendación",
                    value="tab-modelos",
                ),
            ],
        ),
        html.Div(id="tabs-content"),
    ]
)

# ---- Contenido de la pestaña de visión general ----
def layout_tab_general():
    return html.Div(
        [
            html.H3(
                "Información General de los datos históricos",
                style={"textAlign": "center", "marginTop": "30px"},
            ),
            html.Div(
                [
                    html.Div(
                        dcc.Graph(figure=fig_precio_vs_cuartos),
                        style={"width": "33%", "display": "inline-block"},
                    ),
                    html.Div(
                        dcc.Graph(figure=fig_dist_precios),
                        style={"width": "33%", "display": "inline-block"},
                    ),
                    html.Div(
                        dcc.Graph(figure=fig_resp_vs_reco),
                        style={"width": "33%", "display": "inline-block"},
                    ),
                ],
                style={"width": "100%", "textAlign": "center"},
            ),
        ]
    )


# ---- Contenido de la pestaña de modelos ----
def layout_tab_modelos():
    return html.Div(
        [
            html.H4(
                "Ingrese características del Airbnb para estimar el Precio y la Recomendación",
                style={"textAlign": "center", "marginTop": "20px",},
            ),
            html.Div(
                [
                    # =======================
                    # COLUMNA: PREDICCIÓN PRECIO
                    # =======================
                    html.Div(
                        [
                            html.H5("Predicción de Precio"),
                            html.Label("Número de baños (bathroomsf)"),
                            dcc.Input(
                                id="precio-bathroomsf",
                                type="number",
                                value=1,
                                step=0.5,
                                style={"width": "100%"},
                            ),
                            html.Br(),
                            html.Label("Número de cuartos (bedrooms)"),
                            dcc.Input(
                                id="precio-bedrooms",
                                type="number",
                                value=2,
                                step=1,
                                style={"width": "100%"},
                            ),
                            html.Br(),
                            html.Label("Tipo de propiedad (room_type)"),
                            dcc.Dropdown(
                                id="precio-room-type",
                                options=[
                                    {"label": "Entire home/apt", "value": 1},
                                    {"label": "Private room", "value": 0},
                                ],
                                value=1,
                                clearable=False,
                            ),
                            html.Br(),
                            html.Label(
                                "Puntaje de ubicación (review_scores_location) [0-5]"
                            ),
                            dcc.Input(
                                id="precio-review-location",
                                type="number",
                                value=4.5,
                                min=0,
                                max=5,
                                step=0.1,
                                style={"width": "100%"},
                            ),
                            html.Br(),
                            html.Label("¿Tiene jacuzzi? (has_jacuzzi)"),
                            dcc.Dropdown(
                                id="precio-has-jacuzzi",
                                options=[
                                    {"label": "Sí", "value": 1},
                                    {"label": "No", "value": 0},
                                ],
                                value=0,
                                clearable=False,
                            ),
                            html.Br(),
                            html.Label("¿Tiene TV por cable? (has_tv_cable)"),
                            dcc.Dropdown(
                                id="precio-has-tv-cable",
                                options=[
                                    {"label": "Sí", "value": 1},
                                    {"label": "No", "value": 0},
                                ],
                                value=1,
                                clearable=False,
                            ),
                            html.Br(),
                            html.Label("¿Tiene piscina? (has_pool)"),
                            dcc.Dropdown(
                                id="precio-has-pool",
                                options=[
                                    {"label": "Sí", "value": 1},
                                    {"label": "No", "value": 0},
                                ],
                                value=0,
                                clearable=False,
                            ),
                            html.Br(),
                            html.Label(
                                "Ocupación estimada últimos 365 días (estimated_occupancy_l365d)"
                            ),
                            dcc.Input(
                                id="precio-estimated-occupancy",
                                type="number",
                                value=200,
                                min=0,
                                max=365,
                                step=1,
                                style={"width": "100%"},
                            ),
                            html.Br(),
                            html.Label(
                                "Puntaje general (review_scores_rating) [0-5]"
                            ),
                            dcc.Input(
                                id="precio-review-rating",
                                type="number",
                                value=4.6,
                                min=0,
                                max=5,
                                step=0.1,
                                style={"width": "100%"},
                            ),
                            html.Br(),
                            html.Label("Capacidad (accommodates)"),
                            dcc.Input(
                                id="precio-accommodates",
                                type="number",
                                value=4,
                                min=1,
                                step=1,
                                style={"width": "100%"},
                            ),
                            html.Br(),
                            html.Button(
                                "Predecir precio",
                                id="btn-precio",
                                n_clicks=0,
                                style={
                                    "marginTop": "10px",
                                    "width": "100%",
                                },
                            ),
                            html.Div(
                                id="precio-output",
                                style={
                                    "marginTop": "10px",
                                    "padding": "10px",
                                    "backgroundColor": "#f2f2f2",
                                    "textAlign": "center",
                                },
                            ),
                        ],
                        style={"width": "45%", "display": "inline-block"},
                    ),
                    # =======================
                    # COLUMNA: RECOMENDACIÓN
                    # =======================
                    html.Div(
                        [
                            html.H5("Recomendación"),
                            html.Label(
                                "Tasa de respuesta del host (host_response_rate)"
                            ),
                            dcc.Input(
                                id="rec-host-response-rate",
                                type="number",
                                value=0.9,
                                min=0,
                                max=1,
                                step=0.01,
                                style={"width": "100%"},
                            ),
                            html.Br(),
                            html.Label("¿Es superhost? (host_is_superhost)"),
                            dcc.Dropdown(
                                id="rec-superhost",
                                options=[
                                    {"label": "Sí", "value": 1},
                                    {"label": "No", "value": 0},
                                ],
                                value=1,
                                clearable=False,
                            ),
                            html.Br(),
                            html.Label("¿Identidad verificada? (host_identity_verified)"),
                            dcc.Dropdown(
                                id="rec-identity-verified",
                                options=[
                                    {"label": "Sí", "value": 1},
                                    {"label": "No", "value": 0},
                                ],
                                value=1,
                                clearable=False,
                            ),
                            html.Br(),
                            html.Label("¿Reserva instantánea? (instant_bookable)"),
                            dcc.Dropdown(
                                id="rec-instant-bookable",
                                options=[
                                    {"label": "Sí", "value": 1},
                                    {"label": "No", "value": 0},
                                ],
                                value=1,
                                clearable=False,
                            ),
                            html.Br(),
                            html.Label("Noches mínimas (minimum_nights)"),
                            dcc.Input(
                                id="rec-min-nights",
                                type="number",
                                value=2,
                                min=1,
                                step=1,
                                style={"width": "100%"},
                            ),
                            html.Br(),
                            html.Label("Noches máximas (maximum_nights)"),
                            dcc.Input(
                                id="rec-max-nights",
                                type="number",
                                value=30,
                                min=1,
                                step=1,
                                style={"width": "100%"},
                            ),
                            html.Br(),
                            html.Label("Número de baños (bathroomsf)"),
                            dcc.Input(
                                id="rec-bathroomsf",
                                type="number",
                                value=1.5,
                                step=0.5,
                                style={"width": "100%"},
                            ),
                            html.Br(),
                            html.Label("Número de camas (beds)"),
                            dcc.Input(
                                id="rec-beds",
                                type="number",
                                value=2,
                                min=1,
                                step=1,
                                style={"width": "100%"},
                            ),
                            html.Br(),
                            html.Label("Wifi (has_wifi)"),
                            dcc.Dropdown(
                                id="rec-has-wifi",
                                options=[
                                    {"label": "Sí", "value": 1},
                                    {"label": "No", "value": 0},
                                ],
                                value=1,
                                clearable=False,
                            ),
                            html.Br(),
                            html.Label("Aire/Calefacción (has_ac_heating)"),
                            dcc.Dropdown(
                                id="rec-has-ac-heating",
                                options=[
                                    {"label": "Sí", "value": 1},
                                    {"label": "No", "value": 0},
                                ],
                                value=0,
                                clearable=False,
                            ),
                            html.Br(),
                            html.Label("Self check-in (has_self_checkin)"),
                            dcc.Dropdown(
                                id="rec-has-self-checkin",
                                options=[
                                    {"label": "Sí", "value": 1},
                                    {"label": "No", "value": 0},
                                ],
                                value=0,
                                clearable=False,
                            ),
                            html.Br(),
                            html.Label("TV por cable (has_tv_cable)"),
                            dcc.Dropdown(
                                id="rec-has-tv-cable",
                                options=[
                                    {"label": "Sí", "value": 1},
                                    {"label": "No", "value": 0},
                                ],
                                value=0,
                                clearable=False,
                            ),
                            html.Br(),
                            html.Button(
                                "Evaluar recomendación",
                                id="btn-reco",
                                n_clicks=0,
                                style={
                                    "marginTop": "10px",
                                    "width": "100%",
                                },
                            ),
                            html.Div(
                                id="reco-output",
                                style={
                                    "marginTop": "10px",
                                    "padding": "10px",
                                    "backgroundColor": "#f2f2f2",
                                    "textAlign": "center",
                                },
                            ),
                        ],
                        style={
                            "width": "48%",
                            "display": "inline-block",
                            "marginLeft": "4%",
                        },
                    ),
                ]
            ),
        ],style={
            "paddingBottom": "100px",   # espacio visual al final
            "marginBottom": "50px",     # opcional
        }
    )


# ==========================
# CALLBACK TAB PRINCIPAL
# ==========================

@app.callback(Output("tabs-content", "children"), Input("tabs", "value"))
def render_tab_content(tab_value):
    if tab_value == "tab-modelos":
        return layout_tab_modelos()
    return layout_tab_general()


# ==========================
# CALLBACK PREDICCIÓN PRECIO
# ==========================

@app.callback(
    Output("precio-output", "children"),
    Input("btn-precio", "n_clicks"),
    State("precio-bathroomsf", "value"),
    State("precio-bedrooms", "value"),
    State("precio-room-type", "value"),
    State("precio-review-location", "value"),
    State("precio-has-jacuzzi", "value"),
    State("precio-has-tv-cable", "value"),
    State("precio-has-pool", "value"),
    State("precio-estimated-occupancy", "value"),
    State("precio-review-rating", "value"),
    State("precio-accommodates", "value"),
    prevent_initial_call=True,
)
def callback_precio(
    n_clicks,
    bathroomsf,
    bedrooms,
    room_type,
    review_loc,
    has_jacuzzi,
    has_tv_cable,
    has_pool,
    estimated_occupancy,
    review_rating,
    accommodates,
):
    try:
        df_entrada = pd.DataFrame(
            [
                {
                    "bathroomsf": bathroomsf,
                    "bedrooms": bedrooms,
                    "room_type": room_type,
                    "review_scores_location": review_loc,
                    "has_jacuzzi": has_jacuzzi,
                    "has_tv_cable": has_tv_cable,
                    "has_pool": has_pool,
                    "estimated_occupancy_l365d": estimated_occupancy,
                    "review_scores_rating": review_rating,
                    "accommodates": accommodates,
                }
            ]
        )
        precio = predecir_precio(df_entrada)
        return f"Precio estimado: {precio:,.0f}"
    except Exception as e:
        return f"Error al predecir el precio: {e}"


# ==========================
# CALLBACK RECOMENDACIÓN
# ==========================

@app.callback(
    Output("reco-output", "children"),
    Input("btn-reco", "n_clicks"),
    State("rec-host-response-rate", "value"),
    State("rec-superhost", "value"),
    State("rec-identity-verified", "value"),
    State("rec-instant-bookable", "value"),
    State("rec-min-nights", "value"),
    State("rec-max-nights", "value"),
    State("rec-bathroomsf", "value"),
    State("rec-beds", "value"),
    State("rec-has-wifi", "value"),
    State("rec-has-ac-heating", "value"),
    State("rec-has-self-checkin", "value"),
    State("rec-has-tv-cable", "value"),
    prevent_initial_call=True,
)
def callback_recomendacion(
    n_clicks,
    host_response_rate,
    superhost,
    identity_verified,
    instant_bookable,
    min_nights,
    max_nights,
    bathroomsf,
    beds,
    has_wifi,
    has_ac_heating,
    has_self_checkin,
    has_tv_cable,
):
    try:
        df_entrada = pd.DataFrame(
            [
                {
                    "host_response_rate": host_response_rate,
                    "host_is_superhost": superhost,
                    "host_identity_verified": identity_verified,
                    "instant_bookable": instant_bookable,
                    "minimum_nights": min_nights,
                    "maximum_nights": max_nights,
                    "bathroomsf": bathroomsf,
                    "beds": beds,
                    "has_wifi": has_wifi,
                    "has_ac_heating": has_ac_heating,
                    "has_self_checkin": has_self_checkin,
                    "has_tv_cable": has_tv_cable,
                }
            ]
        )
        recomendacion = predecir_recomendable(df_entrada)
        return f"Resultado del modelo: {recomendacion}"
    except Exception as e:
        return f"Error al evaluar la recomendación: {e}"


# ==========================
# MAIN
# ==========================

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=True)
