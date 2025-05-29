import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from threading import Thread
import plotly.graph_objects as go
import json
from websocket import WebSocketApp

app = dash.Dash(__name__)

latest_gps = None
latest_winch = None
gps_path = []
winch_path = []

app.layout = html.Div([
    dcc.Interval(id='update', interval=1000, n_intervals=0),
    dcc.Graph(id='graph3d')
])

app.layout = html.Div([
    dcc.Interval(id='update', interval=1000, n_intervals=0),
    html.H3("SeaHawk", style={'textAlign': 'center', 'fontSize': '3em'}),
    html.Div([
        html.Div(dcc.Graph(id='graph3d', style={'height': '80vh', 'width': '100%'}), style={'flex': '50%'}),
        html.Div(html.Video(src="/assets/winch.mp4", controls=True, loop=True, muted=True,
                            style={'height': '80vh', 'width': '100%'}), style={'flex': '50%'})
    ], style={
        'display': 'flex',
        'justifyContent': 'center',
        'alignItems': 'center',
        'width': '100%',
        'height': '100vh'
    })
], style={'margin': '0', 'padding': '0', 'height': '100vh', 'width': '100vw'})

def handle_gps(msg):
    global latest_gps
    latest_gps = {"x": msg['point']['x'], "y": msg['point']['y'], "z": msg['point']['z']}

def handle_winch(msg):
    global latest_winch
    latest_winch = {"x": msg['point']['x'], "y": msg['point']['y'], "z": msg['point']['z']}

def make_listener(url, topic, handler):
    def on_message(ws, message):
        data = json.loads(message)
        if "msg" in data:
            handler(data["msg"])
    def on_open(ws):
        ws.send(json.dumps({"op": "subscribe", "topic": topic}))
    ws = WebSocketApp(url, on_open=on_open, on_message=on_message)
    Thread(target=ws.run_forever, daemon=True).start()

make_listener("ws://127.0.0.1:9091", "/lariat/current_position_reference", handle_gps) # reference
make_listener("ws://127.0.0.1:9091", "/lariat/winch/uav_pos_winch_enu", handle_winch) # position

@app.callback(
    Output("graph3d", "figure"),
    Input("update", "n_intervals")
)
def update_graph(_):
    if latest_gps:
        gps_path.append([latest_gps["x"], latest_gps["y"], latest_gps["z"]])
    if latest_winch:
        winch_path.append([latest_winch["x"], latest_winch["y"], latest_winch["z"]])

    fig = go.Figure()

    if gps_path:
        x, y, z = zip(*gps_path)
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines+markers',
            marker=dict(size=4, color='blue'),
            line=dict(color='blue', width=2),
            name='reference'
        ))

    if winch_path:
        x, y, z = zip(*winch_path)
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines+markers',
            marker=dict(size=4, color='red'),
            line=dict(color='red', width=2),
            name="position"
        ))

    fig.update_layout(scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='z',
        xaxis=dict(range=[-10, 10]),
        yaxis=dict(range=[-10, 10]),
        zaxis=dict(range=[-0.5, 15]),
    ),
    uirevision='constant',
    height=800,  # increases figure size
    margin=dict(l=0, r=0, b=0, t=0)
    )

    return fig

if __name__ == "__main__":
    app.run(debug=True)
