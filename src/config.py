DEFAULT_ALTITUDE = 25 # base altitude for mapping in meters
LOCAL_POSITION_IN_TOPIC = "/dji_osdk_ros/local_position"
GPS_POSITION_IN_TOPIC = "/dji_osdk_ros/gps_position"
RTK_POSITION_IN_TOPIC = "/dji_osdk_ros/rtk_position"
ATTITUDE_IN_TOPIC = "/dji_osdk_ros/attitude"

PILE_PIXEL_POSITION_OUT_TOPIC = "/multispectral/pile_pixel_position"
PILE_GLOBAL_POSITION_OUT_TOPIC = "/multispectral/pile_global_position" # GPS
PILE_ENU_POSITION_OUT_TOPIC = "/multispectral/pile_enu_position"
DETECTION_IMAGE_OUT_TOPIC =  "/multispectral/detection_image"
TRIGGER_OUT_TOPIC = "/camera/trigger"


LAST_PHOTO_RTK_POSITION_OUT_TOPIC = "/multispectral/last_photo_rtk_position"
LAST_PHOTO_LOCAL_POSITION_OUT_TOPIC = "/multispectral/last_photo_local_position"
LAST_PHOTO_ATTITUDE_OUT_TOPIC = "/multispectral/last_photo_attitude"