from dash import dcc, html
import plotly.graph_objs as go


def render_tab(train_x, train_y, valid_x, valid_y, label, actual_id, predicted_id):
    return dcc.Tab(
        label=f"{label}",
        children=[
            html.Div(
                [
                    html.H2("Actual closing price", style={"textAlign": "center"}),
                    dcc.Graph(
                        id=actual_id,
                        figure={
                            "data": [go.Scatter(x=train_x, y=train_y, mode="markers")],
                            "layout": go.Layout(
                                title="scatter plot",
                                xaxis={"title": "Date"},
                                yaxis={"title": "Closing Rate"},
                            ),
                        },
                    ),
                    html.H2(
                        "LSTM Predicted closing price", style={"textAlign": "center"}
                    ),
                    dcc.Graph(
                        id=predicted_id,
                        figure={
                            "data": [go.Scatter(x=valid_x, y=valid_y, mode="markers")],
                            "layout": go.Layout(
                                title="scatter plot",
                                xaxis={"title": "Date"},
                                yaxis={"title": "Closing Rate"},
                            ),
                        },
                    ),
                ]
            )
        ],
    )
