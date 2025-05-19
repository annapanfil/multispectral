import os 
import exiftool
import folium
import rosbag
import numpy as np

if __name__ == "__main__":
    img_dir = "/home/anna/Datasets/raw_images/hamburg_2025_05_19/images/0000SET"
    panel_img = "0000"
    bag_path = "/home/anna/Datasets/annotated/hamburg_mapping/bags/matriceBag_multispectral_2025-05-19-10-51-49.bag"
    img_topic_name = "/camera/trigger"
    gps_topic_name = "/dji_osdk_ros/gps_position"

    coords = []

    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[gps_topic_name]):
            coords.append([msg.latitude, msg.longitude])
                    
    coords = np.array(coords)
    print(np.mean(coords, axis=0))
    map = folium.Map(location=list(np.mean(coords, axis=0)), zoom_start=18, max_zoom=19)
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        name="Satellite"
        ).add_to(map)

    folium.LayerControl().add_to(map)

    folium.PolyLine(coords, color="red", weight=3).add_to(map)
    map.save('../out/mapping_hamburg.html')