#!/usr/bin/python

import os
import roslib
import sys
import math
import numpy as np
import cv2
import yaml
import rospy
import euler_zyx as ez
import sensor_msgs.point_cloud2 as pc2
import tf_conversions.posemath as pm
import tf
import PyKDL
from message_filters import ApproximateTimeSynchronizer, Subscriber
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from cv_bridge import CvBridge, CvBridgeError

class label_generator:
    def __init__(self):
        cam_intr_fname = rospy.get_param("~cam_intrinsics_yaml")
        cam_lidar_ext_fname = rospy.get_param("~cam_lidar_extrinsics_yaml")
        self.data_folder = rospy.get_param("~data_folder", "/tmp/images")
        self.label_folder = rospy.get_param("~label_folder", "/tmp/labels")
        self.skip_num = rospy.get_param("~skip_num", 1)
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
        if not os.path.exists(self.label_folder):
            os.makedirs(self.label_folder)

        self.height_threshold = rospy.get_param("~height_threshold", 0.0)
        self.bridge = CvBridge()
        self.image_sub = Subscriber("/image_raw", Image)
        self.pose_sub = Subscriber("/current_pose", PoseStamped)
        self.ptcloud_sub = Subscriber("/velodyne_points", PointCloud2)
        self.app_sync = ApproximateTimeSynchronizer( \
            [self.image_sub, self.pose_sub, self.ptcloud_sub], 100, 0.1)
        self.app_sync.registerCallback(self.synced_callback)
        self.img_list = []
        self.pose_list = []
        self.ptcloud_list = []
        self.T_cam2base = None
        self.tf_listener = tf.TransformListener()
        self.counter = 0

        with open(cam_intr_fname, 'r') as stream:
            try:
                cam_intr_yaml = yaml.load(stream)
                cam_intr_data = cam_intr_yaml['camera_matrix']['data']
                cam_dist_data = cam_intr_yaml['distortion_coefficients']['data']
                cam_intrinsics = np.array(cam_intr_data);
                self.image_width = cam_intr_yaml['image_width'];
                self.image_height = cam_intr_yaml['image_height'];
                self.calib_mat = cam_intrinsics.reshape(3, 3)
                self.cam_distortion = np.array(cam_dist_data);
                self.label = np.zeros((self.image_height, self.image_width, 1), dtype = "uint8")
            except yaml.YAMLError as exc:
                print(exc)

        with open(cam_lidar_ext_fname, 'r') as stream:
            try:
                cam_lidar_ext_yaml = yaml.load(stream)
                cam_lidar_ext_data = cam_lidar_ext_yaml['CameraExtrinsicMat']['data']
                cam_lidar_extrinsics = np.array(cam_lidar_ext_data)
                cam_lidar_extrinsics = cam_lidar_extrinsics.reshape(4, 4)
                self.T_lidar2cam = cam_lidar_extrinsics[0:3, 0:4]
                self.T_cam2lidar = np.linalg.inv(cam_lidar_extrinsics)[0:3, 0:4]
                self.t_cam2lidar = self.T_cam2lidar[0:3, 3]
                self.R_cam2lidar = self.T_cam2lidar[0:3, 0:3].dot(ez.eulerZYX(0.04, 0.02, 0.0))
            except yaml.YAMLError as exc:
                print(exc)


    def back_project(self, pt):
        pix = self.calib_mat.dot(pt)
        if pix[2] <= 0:
            return None

        pix = pix/pix[2]
        u = int(pix[0])
        v = int(pix[1])
        return u, v


    # assuming ENU, with x pointing forward in the body frame of the car
    def compute_lidar_label(self, img, pt_gen):
        for p in pt_gen:
            if p[2] < self.height_threshold:
                continue
            pt_cam = self.R_cam2lidar.dot(np.array([p[0], p[1], p[2]])) + self.t_cam2lidar
            pixel = self.back_project(pt_cam)

            if self.in_image(pixel):
                img[0:pixel[1]+20, pixel[0]-20:pixel[0]+20, 0] = 2


    def compute_pose_label(self, img):
        img.fill(0)
        kdl_origin_pose = pm.fromMsg(self.pose_list[0].pose)

        prev_pixel_left = [150, self.image_height - 1]
        prev_pixel_right = [530, self.image_height - 1]
        for pose in self.pose_list:
            kdl_pose = pm.fromMsg(pose.pose)
            delta_pose = kdl_origin_pose.Inverse()*kdl_pose
            pt = np.array([delta_pose.p.x(), delta_pose.p.y(), delta_pose.p.z() - 0.3])
            pt_cam = self.R_cam2base.dot(pt) + self.t_cam2base
            pixel = self.back_project(pt_cam)

            if self.in_image(pixel):

                pt_left = np.array([delta_pose.p.x(), delta_pose.p.y() + 1, delta_pose.p.z() - 0.3])
                pt_left_cam = self.R_cam2base.dot(pt_left) + self.t_cam2base
                pixel_left = self.back_project(pt_left_cam)

                pt_right = np.array([delta_pose.p.x(), delta_pose.p.y() - 1, delta_pose.p.z() - 0.3])
                pt_right_cam = self.R_cam2base.dot(pt_right) + self.t_cam2base
                pixel_right = self.back_project(pt_right_cam)

                contour = np.array([pixel_left, pixel_right, prev_pixel_right, prev_pixel_left])
                cv2.fillPoly(img, np.int32([contour]), (1, 0, 0))
                prev_pixel_left  = pixel_left
                prev_pixel_right = pixel_right



    def in_image(self, pixel):
        return (pixel and pixel[0] < self.image_width and pixel[0] >= 0 and \
            pixel[1] < self.image_height and pixel[1] >= 0)


    def pose_callback(self, pose):
        if self.T_cam2base is None:
            try:
                tfpose = self.tf_listener.lookupTransform('/camera', '/base_link', rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.loginfo("failed to find tf between base_link and camera")
                return
            self.T_cam2base = pm.toMatrix(pm.fromTf(tfpose))
            self.t_cam2base = self.T_cam2base[0:3, 3]
            self.R_cam2base = self.T_cam2base[0:3, 0:3].dot(ez.eulerZYX(0.05, -0.13, 0))
        self.compute_pose_label(self.label)


    def img_callback(self, img):
        try:
            self.image = self.bridge.imgmsg_to_cv2(img, "bgr8")
        except CvBridgeError as e:
            print(e)


    def ptcloud_callback(self, ptcloud):
        if self.image is None:
            return
        gen = pc2.read_points(ptcloud, skip_nans=True, field_names=("x", "y", "z"))

        self.compute_lidar_label(self.label, gen)


    def synced_callback(self, image, pose, ptcloud):
        self.img_list.append(image)
        self.pose_list.append(pose)
        self.ptcloud_list.append(ptcloud)
        if len(self.img_list) < 500:
            return

        self.img_callback(self.img_list[0])
        self.pose_callback(self.pose_list[0])
        self.ptcloud_callback(self.ptcloud_list[0])
        resized_img = cv2.resize(self.image, (480, 360), interpolation=cv2.INTER_LINEAR)
        resized_label = cv2.resize(self.label, (480, 360), interpolation=cv2.INTER_NEAREST)

        if self.counter % self.skip_num == 0:
            img_name = os.path.join("%08d.jpg"%self.counter)
            cv2.imwrite(os.path.join(self.data_folder, img_name), resized_img)
            label_name = os.path.join("%08d_label.jpg"%self.counter)
            cv2.imwrite(os.path.join(self.label_folder, label_name), resized_label)
        self.counter = self.counter + 1

        self.img_list.pop(0)
        self.pose_list.pop(0)
        self.ptcloud_list.pop(0)



def main(args):
    rospy.init_node('label_generator', anonymous=True)
    lg = label_generator()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        if lg.f:
            lg.f.close()

if __name__ == '__main__':
    main(sys.argv)
