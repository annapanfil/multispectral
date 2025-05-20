import signal
import sys
import threading
import rospy
from sensor_msgs.msg import NavSatFix
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_leaflet as dl

gps_topic_name = "/dji_osdk_ros/gps_position"
map_start = [53.470129, 9.984008] # Hamburg drone starting point
latest_gps = {"lat": map_start[0], "lon": map_start[1]}
video1_path = "/assets/video_gnd.mp4"
video2_path = "/assets/hamburg_mapping.mp4"

def gps_callback(msg):
    latest_gps["lat"] = msg.latitude
    latest_gps["lon"] = msg.longitude

def ros_loop():
    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        rate.sleep()


def signal_handler(sig, frame):
    print('Exiting...')
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    rospy.init_node('gps_map_show', anonymous=True, disable_signals=True)
    rospy.Subscriber(gps_topic_name, NavSatFix, gps_callback)
    
    app = dash.Dash(__name__)

    app.layout = html.Div([
        dl.Map(center=map_start, zoom=18, id="map", children=[
            dl.TileLayer(url="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"),
            dl.Marker(id="marker", position=map_start),
            dl.Polyline(id="trail", positions=[])
        ], style={'width': '100%', 'height': '600px'}),
        html.Div([
            html.Video(src=video1_path, controls=True, width="56%", autoPlay=False, loop=True, muted=True),
            html.Video(src=video2_path, controls=True, width="42%", autoPlay=False, loop=True, muted=True),
        ], style={'display': 'flex', 'alignItems': 'flex-start', 'justifyContent': 'space-between', 'marginTop': '20px'}),
        dcc.Interval(id="interval", interval=300, n_intervals=0),
        dcc.Store(id="path", data=[])
    ])
        
    @app.callback(
        Output("marker", "position"),
        Output("map", "center"),
        Output("trail", "positions"),
        Output("path", "data"),
        Input("interval", "n_intervals"),
        State("path", "data")
    )
    def update(n, path):
        lat, lon = latest_gps["lat"], latest_gps["lon"]
        path = path + [[lat, lon]]
        return [lat, lon], [lat, lon], path, path

    app.run(debug=True)