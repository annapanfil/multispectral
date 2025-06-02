import dash
from dash import html, dcc
import dash_leaflet as dl
from dash.dependencies import Output, Input
from threading import Thread
import json
from websocket import WebSocketApp

app = dash.Dash(__name__)

# Globals to store latest GPS data and paths
latest_gps1 = None
latest_gps2 = None
gps1_path = []
gps2_path = []
map_start = [53.470129, 9.984008] # Hamburg drone starting point

#10.0.15.14 ant:3 blue: 50, red:60 

app.layout = html.Div([
    dcc.Store(id='gps1_store'), 
    dcc.Store(id='gps2_store'),
    dcc.Interval(id='update', interval=1000, n_intervals=0),
    html.H3("SeaBees", style={'textAlign': 'center', 'fontSize': '3em'}),
    html.Div([
        dl.Map(center=map_start, zoom=18, id="map", style={"height": "500px", "width": "50%"}),
        html.Video(src="/assets/apads_2_litter.mp4", controls=True, loop=True, muted=True, style={"width": "50%", "marginLeft": "20px"})
        ], style={'display': 'flex', 'justifyContent': 'center', 'gap': '20px'}),
        html.Div([
            html.Video(src="/assets/apads_seacat.mp4", controls=True, loop=True, muted=True, style={"width": "50%"})
        ], style={'textAlign': 'center', 'marginTop': '20px'})

])

def handle_gps1(msg):
    global latest_gps1
    latest_gps1 = {"latitude": msg['latitude'], "longitude": msg['longitude']}

def handle_gps2(msg):
    global latest_gps2
    latest_gps2 = {"latitude": msg['latitude'], "longitude": msg['longitude']}

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

# Replace with your rosbridge websocket URLs and topics
make_listener("ws://127.0.0.1:9091", "/mljet/gps", handle_gps1) #red
make_listener("ws://127.0.0.1:9091", "/losinj/gps", handle_gps2) #blue

@app.callback(
    Output("map", "children"),
    Input("update", "n_intervals")
)
def update_map(_):
    global gps1_path, gps2_path
    children = [dl.TileLayer(url="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}")]

    if latest_gps1:
        gps1_path.append([latest_gps1["latitude"], latest_gps1["longitude"]])
        children.append(dl.Polyline(positions=gps1_path, color="blue"))
        children.append(dl.CircleMarker(center=gps1_path[-1], radius=5, color="blue",  fillColor="blue", fillOpacity=1))

    if latest_gps2:
        gps2_path.append([latest_gps2["latitude"], latest_gps2["longitude"]])
        children.append(dl.Polyline(positions=gps2_path, color="red"))
        children.append(dl.CircleMarker(center=gps2_path[-1], radius=5, color="red",  fillColor="red", fillOpacity=1))

    return children

if __name__ == "__main__":
    app.run(debug=True)
