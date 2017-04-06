cmake_minimum_required(VERSION 2.8.3)
project(visual_dnn_planner)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  geometry_msgs
  cv_bridge
)

catkin_package(
  LIBRARIES visual_dnn_planner
  CATKIN_DEPENDS rospy sensor_msgs cv_bridge geometry_msgs
)

install(PROGRAMS
  scripts/label_generator.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

