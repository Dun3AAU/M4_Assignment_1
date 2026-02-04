import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from dash import Dash, dcc, html, Input, Output, State, dash_table
import plotly.express as px
import plotly.graph_objects as go


# ------------------------
# Core model functions
# ------------------------
def nn(x, weight):
    return x * weight


def loss(y, y_pred):
    return np.mean((y - y_pred) ** 2)


def gradient(x, y, y_pred):
    return np.mean(2 * x * (y_pred - y))


def update_weight(weight, grad, learning_rate):
    return weight - learning_rate * grad


# ------------------------
# Data + training
# ------------------------
def load_data():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/aaubs/ds-master/main/data/Swedish_Auto_Insurance_dataset.csv"
    )
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)
    df = pd.DataFrame(data_scaled, columns=df.columns)
    return df


def train(df, mode="batch", epochs=5, learning_rate=2.0, weight=10.0):
    X_all = df["X"].to_numpy()
    Y_all = df["Y"].to_numpy()

    history = []

    y_pred0 = nn(X_all, weight)
    loss0 = loss(Y_all, y_pred0)
    grad0 = gradient(X_all, Y_all, y_pred0)

    history.append({"epoch": 0, "loss": loss0, "weight": weight, "grad": grad0})

    for epoch in range(1, epochs + 1):
        if mode == "batch":
            y_pred = nn(X_all, weight)
            current_loss = loss(Y_all, y_pred)
            grad = gradient(X_all, Y_all, y_pred)
            weight = update_weight(weight, grad, learning_rate)
        else:
            for x, y in zip(X_all, Y_all):
                y_pred = nn(x, weight)
                current_loss = loss(y, y_pred)
                grad = gradient(x, y, y_pred)
                weight = update_weight(weight, grad, learning_rate)

        history.append({"epoch": epoch, "loss": current_loss, "weight": weight, "grad": grad})

    return pd.DataFrame(history)


# ------------------------
# Loss(w) curve + optimum
# ------------------------
def loss_curve(df, w_min=-5, w_max=15, n_points=300):
    X = df["X"].to_numpy()
    Y = df["Y"].to_numpy()

    w_values = np.linspace(w_min, w_max, n_points)
    losses = []

    for w in w_values:
        y_pred = nn(X, w)
        losses.append(loss(Y, y_pred))

    losses = np.array(losses)
    best_idx = int(np.argmin(losses))
    w_star = float(w_values[best_idx])
    loss_star = float(losses[best_idx])

    curve_df = pd.DataFrame({"w": w_values, "loss": losses})
    return curve_df, w_star, loss_star


# ------------------------
# Dash app
# ------------------------
app = Dash(__name__)
df = load_data()

app.layout = html.Div(
    style={
        "width": "98vw",
        "maxWidth": "1600px",
        "margin": "10px auto",
        "padding": "10px",
        "fontFamily": "Arial",
    },
    children=[
        html.H2("1-Weight Model Training Dashboard (Dash)", style={"marginBottom": "10px"}),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 320px",
                "gap": "16px",
                "alignItems": "start",
            },
            children=[
                # ==========================
                # LEFT SIDE (PLOTS + TABLE)
                # ==========================
                html.Div(
                    children=[
                        # Top row (wide important plots)
                        html.Div(
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "1fr 1fr",
                                "gap": "16px",
                            },
                            children=[
                                dcc.Graph(id="data-fit-plot", style={"height": "420px"}),
                                dcc.Graph(id="loss-w-plot", style={"height": "420px"}),
                            ],
                        ),

                        # Bottom row (epoch plots)
                        html.Div(
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "1fr 1fr",
                                "gap": "16px",
                                "marginTop": "8px",
                            },
                            children=[
                                dcc.Graph(id="loss-plot", style={"height": "320px"}),
                                dcc.Graph(id="weight-plot", style={"height": "320px"}),
                            ],
                        ),

                        html.Div(style={"marginTop": "10px"}, children=[
                            html.H4("Training history (last rows)", style={"marginBottom": "6px"}),
                            dash_table.DataTable(
                                id="history-table",
                                page_size=10,
                                style_table={"overflowX": "auto"},
                                style_cell={
                                    "textAlign": "left",
                                    "fontFamily": "Arial",
                                    "fontSize": "13px",
                                    "padding": "6px",
                                },
                                style_header={"fontWeight": "bold"},
                            ),
                        ]),
                    ]
                ),

                # ==========================
                # RIGHT SIDE (CONTROLS)
                # ==========================
                html.Div(
                    style={
                        "padding": "14px",
                        "border": "1px solid #ddd",
                        "borderRadius": "14px",
                        "position": "sticky",
                        "top": "10px",
                    },
                    children=[
                        html.H4("Controls", style={"marginTop": "0px"}),

                        html.Label("Mode"),
                        dcc.Dropdown(
                            id="mode",
                            options=[
                                {"label": "Batch", "value": "batch"},
                                {"label": "Stochastic", "value": "stochastic"},
                            ],
                            value="batch",
                            clearable=False,
                        ),

                        html.Br(),
                        html.Label("Epochs"),
                        dcc.Slider(
                            id="epochs",
                            min=1,
                            max=50,
                            step=1,
                            value=5,
                            marks={1: "1", 10: "10", 25: "25", 50: "50"},
                        ),

                        html.Br(),
                        html.Label("Learning rate"),
                        dcc.Input(
                            id="lr",
                            type="number",
                            value=2.0,
                            step=0.1,
                            style={"width": "100%"},
                        ),

                        html.Br(),
                        html.Br(),
                        html.Label("Initial weight"),
                        dcc.Input(
                            id="w0",
                            type="number",
                            value=10.0,
                            step=0.5,
                            style={"width": "100%"},
                        ),

                        html.Br(),
                        html.Br(),

                        # Smaller curve range inputs (inline)
                        html.Label("Loss curve range (w)"),
                        html.Div(
                            style={"display": "flex", "gap": "8px"},
                            children=[
                                dcc.Input(
                                    id="wmin",
                                    type="number",
                                    value=-5,
                                    step=1,
                                    style={"width": "50%"},
                                ),
                                dcc.Input(
                                    id="wmax",
                                    type="number",
                                    value=15,
                                    step=1,
                                    style={"width": "50%"},
                                ),
                            ],
                        ),

                        html.Br(),
                        html.Button(
                            "Run training",
                            id="run",
                            n_clicks=0,
                            style={
                                "width": "100%",
                                "padding": "10px",
                                "borderRadius": "10px",
                                "cursor": "pointer",
                            },
                        ),

                        html.Div(id="summary", style={"marginTop": "12px", "fontSize": "14px"}),
                    ],
                ),
            ],
        ),
    ],
)


@app.callback(
    Output("loss-plot", "figure"),
    Output("weight-plot", "figure"),
    Output("data-fit-plot", "figure"),
    Output("loss-w-plot", "figure"),
    Output("history-table", "data"),
    Output("history-table", "columns"),
    Output("summary", "children"),
    Input("run", "n_clicks"),
    State("mode", "value"),
    State("epochs", "value"),
    State("lr", "value"),
    State("w0", "value"),
    State("wmin", "value"),
    State("wmax", "value"),
)
def update_dashboard(n_clicks, mode, epochs, lr, w0, wmin, wmax):
    epochs = int(epochs)
    lr = float(lr)
    w0 = float(w0)
    wmin = float(wmin)
    wmax = float(wmax)

    hist = train(df, mode=mode, epochs=epochs, learning_rate=lr, weight=w0)
    final_w = float(hist.iloc[-1]["weight"])

    # Loss vs Epoch
    fig_loss = px.line(hist, x="epoch", y="loss", title="Loss vs Epoch")
    fig_loss.update_layout(margin=dict(l=30, r=10, t=40, b=30))

    # Weight vs Epoch
    fig_weight = px.line(hist, x="epoch", y="weight", title="Weight vs Epoch")
    fig_weight.update_layout(margin=dict(l=30, r=10, t=40, b=30))

    # Dataset + model fit
    df_plot = df.copy()
    df_plot["Y_pred"] = nn(df_plot["X"], final_w)

    fig_data = go.Figure()
    fig_data.add_trace(
        go.Scatter(
            x=df_plot["X"],
            y=df_plot["Y"],
            mode="markers",
            name="Data (X,Y)",
        )
    )
    fig_data.add_trace(
        go.Scatter(
            x=df_plot["X"],
            y=df_plot["Y_pred"],
            mode="lines",
            name=f"Model: Y_pred = X * w (w={final_w:.4f})",
        )
    )
    fig_data.update_layout(
        title="Dataset + Model Fit",
        xaxis_title="X",
        yaxis_title="Y",
        margin=dict(l=30, r=10, t=40, b=30),
    )

    # Loss(w) curve
    curve_df, w_star, loss_star = loss_curve(df, w_min=wmin, w_max=wmax, n_points=400)

    fig_loss_w = px.line(curve_df, x="w", y="loss", title="Loss(w) curve (static for dataset)")
    fig_loss_w.add_trace(
        go.Scatter(
            x=[w_star],
            y=[loss_star],
            mode="markers",
            name=f"Optimal w* = {w_star:.4f}",
            marker=dict(size=12),
        )
    )

    current_loss_at_final_w = float(loss(df["Y"], nn(df["X"], final_w)))
    fig_loss_w.add_trace(
        go.Scatter(
            x=[final_w],
            y=[current_loss_at_final_w],
            mode="markers",
            name=f"Current w = {final_w:.4f}",
            marker=dict(size=12),
        )
    )
    fig_loss_w.update_layout(margin=dict(l=30, r=10, t=40, b=30))

    # Table (last rows)
    table_df = hist.tail(15).copy()
    table_df["loss"] = table_df["loss"].round(6)
    table_df["weight"] = table_df["weight"].round(6)
    table_df["grad"] = table_df["grad"].round(6)

    data = table_df.to_dict("records")
    columns = [{"name": c, "id": c} for c in table_df.columns]

    summary = html.Div(
        children=[
            html.Div(f"Final weight: {final_w:.6f}"),
            html.Div(f"Optimal w*: {w_star:.6f}"),
            html.Div(f"Loss at final w: {current_loss_at_final_w:.6f}"),
            html.Div(f"Loss at w*: {loss_star:.6f}"),
        ]
    )

    return fig_loss, fig_weight, fig_data, fig_loss_w, data, columns, summary


if __name__ == "__main__":
    app.run(debug=True)
