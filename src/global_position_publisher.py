#!/usr/bin/env python

import rospy
import math
import numpy as np
import cv2
import sys
import src.gps_utils as gps_utils
from src.config import RTK_POSITION_IN_TOPIC, ATTITUDE_IN_TOPIC, LOCAL_POSITION_IN_TOPIC, PILE_ENU_POSITION_OUT_TOPIC, PILE_GLOBAL_POSITION_OUT_TOPIC, PILE_PIXEL_POSITION_OUT_TOPIC
from geometry_msgs.msg import Point,Vector3Stamped,PointStamped, QuaternionStamped
from std_msgs.msg import Float64
from sensor_msgs.msg import NavSatFix
from src.processing.consts import CAMERA_MATRIX, DISTORTION_COEFFS
from tf.transformations import euler_from_quaternion, euler_from_matrix


class CameraModel:
    def __init__(self, camera_matrix, distortion_coefs):
        """
            Args:
                camera_matrix : [1661.03506282, 0.000000, 960.61997498, 0.000000, 1659.21818332, 536.34363925, 0.000000, 0.000000, 1.000000]
                distortion_coefs: [-0.00344134, -0.00067102, -0.00087853, 0.00110982, 0.000000]

        """
        
        self.camera_matrix = np.reshape(np.array(camera_matrix,dtype=np.float32), (3,3))
        self.distortion_coefs = np.array(distortion_coefs,dtype=np.float32)

        self._ned_to_enu = lambda ned: (ned[1], ned[0], -ned[2]) # converts ned to enu vector
        self.camera_cord_pub = rospy.Publisher("/matrice/obj_camera", PointStamped, queue_size = 30) # ROV position in camera coord. frame
        
    def print_model_info(self):
        print(self.cam_values)
    
    def _camera_to_ned(self, roll, pitch, yaw):
        '''
            Transforms camera (Z forward looking) to NED coordinates
            Args : 
                roll,pitch,yaw - matrice reported in NED frame
        '''
        cam_to_gimbal = self.get_rotation_matrix(roll,pitch,yaw) # cam to gimbal_link(NED)
        return cam_to_gimbal

    def _camera_to_enu(self, drone_orientation_enu):
        '''
            Transforms camera (Z forward looking) to ENU coordinates
            Args : 
                roll,pitch,yaw - drone_orientation in ENU coordinate frame (rpy) 
        '''
        roll,pitch,yaw = drone_orientation_enu
        drone_to_enu = self.get_rotation_matrix(roll,pitch,yaw) # ENU CF -> Drone FLU CF
        cam_to_drone = self.get_rotation_matrix(-180,0,-90) # FLU CF -> cam CF (z forward)
        roll,pitch,yaw = euler_from_matrix(np.matmul(drone_to_enu, cam_to_drone), "sxyz") # check if the rotation is correct
        return np.matmul(drone_to_enu, cam_to_drone)

    def _ned_to_camera(self, roll,pitch,yaw):
        '''
            Transforms NED to camera coordinates
            Args : 
                roll,pitch,yaw - matrice reported camera orientation in NED frame
        '''
        return np.linalg.inv(self._camera_to_ned(roll,pitch,yaw))

    # def camera_to_enu(self, obj_cam, camera_orientation):
    #     # Convert back from camera to gimbal ENU frame (cam->NED->ENU)
    #     cam_r,cam_p,cam_y = camera_orientation
    #     R_ = self._camera_to_ned(cam_r,cam_p,cam_y)
    #     obj_gimbal_ned = np.matmul(R_, obj_cam) # Object in gimbal link origin NED frame
    #     obj_gimbal_enu = np.array(self._ned_to_enu(obj_gimbal_ned)).reshape((3,1))  # Object in gimbal link origin ENU frame
    #     return obj_gimbal_enu

    def _publish_camera_coord(self, coord):
        point_msg = PointStamped()
        point_msg.header.frame_id = "matrice_camera"
        point_msg.header.stamp = rospy.Time.now() # TODO rov_px stamp
        point_msg.point.x = coord[0]
        point_msg.point.y = coord[1]
        point_msg.point.z = coord[2]
        self.camera_cord_pub.publish(point_msg)
    

    def get_obj_camera(self, obj_px, camera_orientation_ned, height, height_offset):
        '''
            Args:
                obj_px - (u,v) pixel coordinates of object
                camera_orientation_ned - (roll,pitch,yaw) matrice reported camera orientation in NED frame
                height - height of camera above ground plane
            Returns:
                obj_cam - object surface position
        '''
        obj_norm_cam = self.px_to_norm_cam(obj_px) # Normalized camera coordinates of object
        ## Fetch NED to Camera coordinate transform
        cam_r,cam_p,cam_y = camera_orientation_ned
        R_ = self._ned_to_camera(cam_r,cam_p,cam_y)
        plane_normal = np.array([0, 0, 1]).reshape(3,1) # normal in gimbal_link (NED frame)
        plane_normal_cam = np.matmul(R_, plane_normal)
        # Define object plane in NED coordinates (gimbal_link origin)
        print(f"{height=} + {height_offset}")
        plane_p0 = np.array([0, 0, height+height_offset]).reshape(3,1) #point on ROV plane in gimbal_link (NED frame)
        plane_p0_cam = np.matmul(R_, plane_p0)
        # Define offset plane in NED coordinates (origin on surface)
        # intersect object camera coordinates and ROV plane
        obj_cam = self._intersect_vector_plane(obj_norm_cam, plane_p0_cam, plane_normal_cam) #object in camera coordinates

        #Publish to topics
        self._publish_camera_coord(obj_cam) # Publish surface camera coordinates
        return obj_cam

    def _intersect_vector_plane(self, vect, plane_p0, plane_normal):
        '''
            Returns point at which vector intersects the plane
        '''
        d = np.matmul(plane_p0.transpose(), plane_normal) / np.matmul(vect.transpose(),plane_normal)
        intersect_p = d * vect
        return intersect_p

    def px_to_norm_cam(self, obj_px):
        '''
            Convert pixel coordinates to normalized camera coordinates by inverting the calibration matrix and undistorting the input
            Args:
                obj_px - (u,v) pixel coordinates of object
        '''
        u,v = obj_px
        undist_input = np.array([u,v]).reshape((1,1,2)) # reshaped input to work with undistortPoints OpenCV function
        x_ = cv2.undistortPoints(undist_input, self.camera_matrix, self.distortion_coefs) # normalized image coordinates
        x_ = x_.reshape((2,1)) # back to (2,1)
        x_ = np.append(x_,np.array([[1]]),axis=0) # normalized camera coordinates 
        return x_

    def get_rotation_matrix(self, roll, pitch, yaw):
        '''
        '''
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)
        Rx = np.array([[1, 0, 0], [0, math.cos(roll), -1 * math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
        Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-1 * math.sin(pitch), 0, math.cos(pitch)]])
        Rz = np.array([[math.cos(yaw), -1 * math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        return np.matmul(Rz, np.matmul(Ry, Rx))
    

class ObjectGlobalPositionPublisher:
    def __init__(self, camera_model, geo_ref_pos, topics, height_offset = 0.0, publish_constantly = False):
        '''
            Args:
                topics = {
                    px_cord : (px_cord_topic_name, topic_type)
                    gps: (gps_topic_name, topic_type)
                    gimbal : gimbal_topic_name, topic_type)
                }
                camera_model - CameraModel object to convert from pixel to drone relative coordinates
                rov_px_topic - topic for listening for pixel coordinates of the rov (PointStamped)
                geo_reference_pos - initial drone position GPS coordinates (lat,long,alt)
                publish_constantly - if true, will publish object position constantly, otherwise only when px_cord message arrives
        '''
        self.geo_reference_pos = geo_ref_pos
        self.camera_model = camera_model
        self.height_offset = height_offset
        #Messages, subs and pubs
        self.msgs = {}

        #Subs
        self.px_cord_sub = rospy.Subscriber(topics["px_cord"][0], topics["px_cord"][1], callback=self._px_cord_callback) # Pixel coordinates of object
        self.gps_sub = rospy.Subscriber(topics["gps"][0], topics["gps"][1], callback=self._gps_callback)
        self.drone_att_sub = rospy.Subscriber(topics["attitude"][0], topics["attitude"][1], callback=self._attitude_callback)
        self.height_sub = rospy.Subscriber(topics["height"][0], topics["height"][1], callback=self._height_callback)
        if topics["gimbal"][1]==None:
            self.msgs["gimbal"] = Vector3Stamped()
            self.msgs["gimbal"].vector.x = 0.0
            self.msgs["gimbal"].vector.y = -90.0
            self.msgs["gimbal"].vector.z = 0.0
        else:
            self.gimbal_sub = rospy.Subscriber(topics["gimbal"][0], topics["gimbal"][1], callback=self._gimbal_callback)
        #Pubs
        self.global_pos_pub = rospy.Publisher(PILE_GLOBAL_POSITION_OUT_TOPIC, NavSatFix,queue_size=30) # GPS coordinates of object
        self.world_enu_pub = rospy.Publisher(PILE_ENU_POSITION_OUT_TOPIC, PointStamped, queue_size=30) # ENU positions of object with respect to reference point
        self._ned_to_enu = lambda ned: (ned[1], ned[0], -ned[2]) # converts ned to enu vector
        self.publish_constantly = publish_constantly # if true, will publish object position constantly, otherwise only when px_cord message arrives

        #Initialize geo system
        self.gps_utils = gps_utils.GPS_utils()
        lat0, lon0, alt0 = self.geo_reference_pos
        self.gps_utils.setENUorigin(lat0,lon0,alt0)

    def _msgs_arrived(self):
        '''
            Returns true if all messages arrived
        '''
        keys = self.msgs.keys()
        return "px_cord" in keys and "gps" in keys and "gimbal" in keys and "height" in keys

    def _gps_to_global_pos_enu(self, gps_pos):
        '''
            Calculates GPS position offset from initial (reference) drone position in ENU
            Args:
                gps_pos - current drone gps position (lat,long,alt)
        '''
        lat, lon, alt = gps_pos
        return self.gps_utils.geo2enu(lat, lon, alt)
    
    def _get_camera_orientation_enu(self, drone_orientation_enu):
        R_ = self.camera_model._camera_to_enu(drone_orientation_enu)
        roll, pitch, yaw = euler_from_matrix(R_,"sxyz")
        return roll, pitch, yaw

    def _get_camera_orientation_ned(self, drone_orientation_enu):
        R_ = self.camera_model._camera_to_enu(drone_orientation_enu)
        R_enu_to_ned = self.camera_model.get_rotation_matrix(180.0,0,90.0)
        R_ = np.matmul(R_enu_to_ned, R_)
        roll, pitch, yaw = euler_from_matrix(R_,"sxyz")
        return roll, pitch, yaw

    
    def _get_drone_pos_enu(self):
        gps_pos = (self.msgs["gps"].latitude, self.msgs["gps"].longitude, self.msgs["height"].point.z)
        #print("Drone GPS : {}".format(gps_pos))
        enu_pos = np.squeeze(np.asarray(self._gps_to_global_pos_enu(gps_pos)))
        return enu_pos
    
    def _get_drone_pos_grid(self, grid_pos = (15.35, 6.7)):
        drone_pos = np.array([15.5, 6.7, self.msgs["height"].data+2.0])
        R = self.camera_model.get_rotation_matrix(0.0,0.0,-67.3)
        R_ = np.linalg.inv(R)
        drone_pos_enu = np.matmul(R_,drone_pos) # GroundTruth to ENU
        return drone_pos_enu
    
    def _get_camera_offset(self):
        camera_offset = np.array([0.14, 0.0, -0.25]) # in in FLU
        R_i_b = self.camera_model.get_rotation_matrix(self.msgs["attitude"].vector.x, self.msgs["attitude"].vector.y, self.msgs["attitude"].vector.z)
        camera_offset_enu = np.matmul(R_i_b,camera_offset)
        return camera_offset_enu

    def _get_obj_px_position(self):
        x,y = self.msgs["px_cord"].point.x, self.msgs["px_cord"].point.y # Real stuff for OBJECT TRACKING
        if not self.publish_constantly:
            self.msgs.pop("px_cord", None) # Remove message from dict
        return (x,y)

    def _px_cord_callback(self, msg):
        self.msgs["px_cord"] = msg

    def _height_callback(self, msg):
        self.msgs["height"] = msg

    def _gps_callback(self, msg):
        self.msgs["gps"] = msg

    def _attitude_callback(self,msg):
        att_quaternion = [msg.quaternion.x, msg.quaternion.y, msg.quaternion.z, msg.quaternion.w]
        r_enu,p_enu,y_enu = euler_from_quaternion(att_quaternion)
        r_enu,p_enu,y_enu = math.degrees(r_enu), math.degrees(p_enu), math.degrees(y_enu) # ENU rpy
        #Convert to Vector3Stamped message
        vector_msg = Vector3Stamped()
        vector_msg.header = msg.header
        vector_msg.vector.x, vector_msg.vector.y, vector_msg.vector.z = r_enu, p_enu, y_enu
        self.msgs["attitude"] = vector_msg

    def _gimbal_callback(self, msg):
        self.msgs["gimbal"] = msg
        
    def _pub_gps_position(self, pub, gps_pos, stamp):
        object_lat, object_lon, object_alt = gps_pos
        sat_msg = NavSatFix()
        sat_msg.header.stamp = stamp
        sat_msg.latitude = object_lat
        sat_msg.longitude = object_lon
        sat_msg.altitude = object_alt
        sat_msg.position_covariance_type = 0 #UNKNOWN
        pub.publish(sat_msg)
    
    def _pub_world_enu_pos(self, pub, enu_pos, stamp):
        object_e, object_n, object_u = enu_pos
        point_msg = PointStamped()
        point_msg.header.frame_id = "world_enu"
        point_msg.header.stamp = stamp
        point_msg.point.x = object_e
        point_msg.point.y = object_n
        point_msg.point.z = object_u
        pub.publish(point_msg)
    
    def _get_camera_position_world_enu(self, debug = False):
        '''
            Calculates camera position in world ENU (based on GPS world origin
            set in gps_utils.enu2geo module and drone GPS position )
        '''
        drone_pos = self._get_drone_pos_enu()
        e,n,u = drone_pos[0], drone_pos[1], drone_pos[2]
        drone_pos = np.array((drone_pos[0], drone_pos[1], drone_pos[2]))
        camera_pos = drone_pos + self._get_camera_offset()
        camera_pos_geo = self.gps_utils.enu2geo(camera_pos[0],camera_pos[1],camera_pos[2])
        camera_pos = np.array(camera_pos).reshape((3,1))
        if debug:
            print(f"Drone position ENU world (x,y,z): ({drone_pos[0]}, {drone_pos[1]}, {drone_pos[2]})")
            print("Camera position ENU world (x,y,z): ({}, {}, {})".format(camera_pos[0],camera_pos[1],camera_pos[2]))
            print("Camera position GEO (lat,lon,height): ({}, {}, {})".format(camera_pos_geo[0],camera_pos_geo[1],camera_pos_geo[2]))
        return camera_pos

    def start_publishing(self):
        rate_wait = rospy.Rate(5)
        rate = rospy.Rate(30)
        while not self._msgs_arrived():
            # print(self.msgs.keys())
            if rospy.is_shutdown():
                sys.exit()
            rate_wait.sleep()
            print("Waiting for messages. Have: {}".format(self.msgs.keys()))

        while not rospy.is_shutdown():
            while not self._msgs_arrived():
                if rospy.is_shutdown():
                    sys.exit()
                rate_wait.sleep()
                # print("Waiting for messages. Have: {}".format(self.msgs.keys()))
            stamp = self.msgs["px_cord"].header.stamp
            camera_pos_enu = self._get_camera_position_world_enu()
            obj_px_position = self._get_obj_px_position()
            drone_orientation_enu = (self.msgs["attitude"].vector.x, self.msgs["attitude"].vector.y, self.msgs["attitude"].vector.z)

            camera_orientation_ned = self._get_camera_orientation_ned(drone_orientation_enu)
            obj_cam = self.camera_model.get_obj_camera(obj_px_position, camera_orientation_ned, height = camera_pos_enu[2][0], height_offset=self.height_offset)#
            R_camera_to_enu = self.camera_model._camera_to_enu(drone_orientation_enu)
            obj_enu = np.matmul(R_camera_to_enu, obj_cam) # object in ENU 
            #World coordinates (ENU)
            obj_enu = camera_pos_enu + obj_enu
            #Geodetic coordinates (lat,lon,alt)
            obj_geo = self.gps_utils.enu2geo(obj_enu[0],obj_enu[1],obj_enu[2])
            #Publish coordinates
            self._pub_world_enu_pos(self.world_enu_pub, obj_enu, stamp)
            self._pub_gps_position(self.global_pos_pub, obj_geo, stamp)
            print(f"ENU position: {', '.join([str(x) for x in obj_enu])}\nGPS postion: {', '.join([str(x) for x in obj_geo])}")
            print(f"Waiting for next px_cord message")
            rate.sleep()

if __name__=="__main__":
    rospy.init_node("object_global_position_publisher")
    camera_model = CameraModel(CAMERA_MATRIX, DISTORTION_COEFFS)
    geo_ref_hamburg = (53.470132, 9.984003, 0.0) # Hamburg experiments reference point
    # geo_ref_dubrovnik = (42.66439066892307, 18.071053781363926, 0.0) # Dubrovnik experiments
    height_offset = 2 # from sea level to drone starting point (in meters)

    topics = {
		"gimbal" : ("gimbal", None),
		"gps" : (RTK_POSITION_IN_TOPIC, NavSatFix),
		"px_cord" : (PILE_PIXEL_POSITION_OUT_TOPIC, PointStamped),
        "height" : (LOCAL_POSITION_IN_TOPIC, PointStamped),
        "attitude": (ATTITUDE_IN_TOPIC, QuaternionStamped),
	} 

    glob_pub = ObjectGlobalPositionPublisher(camera_model, geo_ref_hamburg, topics, height_offset)
    glob_pub.start_publishing()
