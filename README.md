# Visual DNN Planner

This repo generates path proposal labels from LIDAR pointclouds, pose estimates and mono images. The labels can be used to train a CNN autoencoder to propose valid path for autonomous car applications. It is based on the the following work:

```
D. Barnes, W. Maddern and I. Posner: "Find Your Own Way: Weakly-Supervised Segmentation of Path Proposals for Urban Autonomy,"
```
Check out preliminary results [here](https://www.youtube.com/watch?v=GW6KPaugSDU&t=45s)!

## Instructions to run
Please modify *camera_lidar_extrinsic.yaml* to reflect the proper extrinsic calibration parameters from the camera to the LIDAR. The file *udacity_sdc_center_cam.yaml* should be changed to values for the intrinsic parameters of your camera.

To launch the label generator against the Udacity ROS bag datasets [here](https://github.com/udacity/self-driving-car/tree/master/datasets), please run:
```
roslaunch visual_dnn_planner label_generator.launch
rosbag play <bag name>.bag
```

The output images are saved under */tmp* by default.

## TODOs
1. Include SegNet CNN training and testing scripts after cleanup
