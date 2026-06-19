%% Define paths for all data files

% original measurement data root folder
data_dir = fullfile("D:\m.von-arnim\Documents\data\CDebeunne\2025_PG_Fusion\building2");
% data_dir = fullfile("D:\m.von-arnim\Documents\data\CDebeunne\2025_PG_Fusion\loops\");

% ublox measurement data location
ublox_bagdir = fullfile("E:\data\meas_max_cesar\building2\");
% ublox_bagdir = fullfile("E:\data\meas_max_cesar\loops\");

% ground truth
GT_file = fullfile(data_dir, "traj_2024_09_30_14_28_36.txt");

% estimator results
sadvio_bagdir = fullfile(data_dir, "\_results\SaDVIO\bags\rosbag2_2026_05_08-18_02_54");

%% Read the recorded pose estimate
rd = ros2bagreader(sadvio_bagdir);
pose_rd = select(rd, "Topic", "/vo_pose");
poses = ros2baggetPoseStamped(pose_rd);

%% Read the ground truth

GT_data = import_data(GT_file);


%% Read the associated ublox GNSS data (if available)
rd_meas = ros2bagreader(ublox_bagdir);
ublox_rd = select(rd_meas, "Topic", "/fix");
pvt_rd = select(rd_meas, "Topic", "/navpvt");
raw_rd = select(rd_meas, "Topic", "/rxmraw");

% ublox = readMessages(ublox_rd);
ublox.navSatFix = ros2baggetNavSatFix(ublox_rd);
ublox.navPvt = ros2baggetubloxNavPVT(pvt_rd);
% ublox.gnssRaw = ros2baggetubloxRxmRAWX()

ublox_enu = lla2enu([ublox.latitude, ublox.longitude, ublox.altitude], ...
    [ublox.latitude(1), ublox.longitude(1), ublox.altitude(1)],"flat");

%% Plot the VIO & ublox ENU position
figure;
scatter(poses_tbl.pose.position.x, poses_tbl.pose.position.y)
hold on
scatter(ublox_enu(:,1), ublox_enu(:,2))
axis equal
title("Position SaDVIO")
xlabel("x [m]")
ylabel("y [m]")