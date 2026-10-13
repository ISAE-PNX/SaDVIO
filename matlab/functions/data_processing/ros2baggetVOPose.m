function [vo_pose] = ros2baggetVOPose(rd)
%ROS2BAGGETVOPOSE Converts a ros2bag from the /vo_pose VIO output to a
%Matlab array
arguments (Input)
    rd  ros2bagreader
end

arguments (Output)
    vo_pose
end
vo_pose = ros2baggetPoseStamped(rd);
end