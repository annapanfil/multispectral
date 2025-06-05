import base64
import cv2
import numpy as np
import dash
from dash import html, dcc
import dash_leaflet as dl
from dash.dependencies import Output, Input
from threading import Thread
import json
from websocket import WebSocketApp

app = dash.Dash(__name__)

# Globals to store latest GPS data and paths
latest_gps = None
gps_path = []
litter_gps = []
latest_img = None
map_start = [53.470129, 9.984008] # Hamburg drone starting point
gps_topic = "/dji_osdk_ros/gps_position"
gps_litter_topic = "/multispectral/pile_global_position"
detection_image_topic = "/multispectral/detection_image"

video1_path = "/assets/multispectral.mp4"
video2_path = "/assets/hamburg_mapping.mp4"
ws_ip = "127.0.0.0"

app.layout = html.Div([
    dcc.Interval(id='update', interval=1000, n_intervals=0),
    # lewy panel z tytułem i mapą
    html.Div([
        html.H3("SeaHawk", style={'textAlign': 'center', 'fontSize': '3em', 'marginBottom': '20px'}),
        dl.Map(center=map_start, zoom=18, id="map", style={"flex": "2", "marginRight": "20px"})
    ], style={"display": "flex", "flexDirection": "column", "flex": "2"}),

    # prawy panel z obrazkiem
    html.Div([
        html.H4("Last detection", style={"textAlign": "center"}),
        html.Img(id='live_image', style={"width": "100%", "height": "auto", "objectFit": "contain"})
    ], style={"flex": "1.5", "display": "flex", "flexDirection": "column", "justifyContent": "center"})
], style={"display": "flex", "height": "100vh", "margin": "0", "padding": "0"})



def handle_gps(msg):
    global latest_gps
    latest_gps = {"latitude": msg['latitude'], "longitude": msg['longitude']}

def handle_litter_gps(msg):
    global litter_gps
    litter_gps.append([msg['latitude'], msg['longitude']])

def handle_image(msg):
    global latest_img
    img_bytes = base64.b64decode(msg["data"])
    img = np.frombuffer(img_bytes, dtype=np.uint8).reshape((msg["height"], msg["width"], -1))
    latest_img = img


def make_listener(url, topic, handler):
    print(f"Connecting to {url} for topic {topic}")
    def on_message(ws, message):
        data = json.loads(message)
        if "msg" in data:
            handler(data["msg"])
    def on_open(ws):
        ws.send(json.dumps({"op": "subscribe", "topic": topic}))
    ws = WebSocketApp(url, on_open=on_open, on_message=on_message)
    Thread(target=ws.run_forever, daemon=True).start()

make_listener(f"ws://{ws_ip}:9091", gps_topic, handle_gps)
make_listener(f"ws://{ws_ip}:9091", gps_litter_topic, handle_litter_gps)
make_listener(f"ws://{ws_ip}:9091", detection_image_topic, handle_image)

@app.callback(
    Output("map", "children"),
    Input("update", "n_intervals")
)
def update_map(_):
    global gps_path
    children = [dl.TileLayer(url="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}")]

    if latest_gps:
        gps_path.append([latest_gps["latitude"], latest_gps["longitude"]])
        children.append(dl.Polyline(positions=gps_path, color="yellow"))
        children.append(dl.CircleMarker(center=gps_path[-1], radius=5, color="yellow",  fillColor="yellow", fillOpacity=1))
        
        for litter in litter_gps:
            children.append(dl.CircleMarker(center=litter, radius=2, color="red",  fillColor="red", fillOpacity=1))

    return children

@app.callback(
    Output('live_image', 'src'),
    Input('update', 'n_intervals')
)
def update_image(_):
    global latest_img
    if latest_img is not None:
        img = latest_img
    else:
        img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    _, buffer = cv2.imencode('.jpg', img)
    encoded = base64.b64encode(buffer).decode('utf-8')
    return "data:image/jpeg;base64," + encoded

if __name__ == "__main__":
    app.run(debug=True)
