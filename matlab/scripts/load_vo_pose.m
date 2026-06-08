data_dir = fullfile("D:\m.von-arnim\Documents\data\CDebeunne\2025_PG_Fusion\building2");
% data_dir = fullfile("D:\m.von-arnim\Documents\data\CDebeunne\2025_PG_Fusion\loops\");
ublox_bagdir = fullfile("E:\data\meas_max_cesar\building2\");
% ublox_bagdir = fullfile("E:\data\meas_max_cesar\loops\");

GT_file = fullfile(data_dir, "traj_2024_09_30_14_28_36.txt");
sadvio_bagdir = fullfile(data_dir, "\_results\SaDVIO\bags\rosbag2_2026_05_08-18_02_54");

%%
rd = ros2bagreader(sadvio_bagdir);
pose_rd = select(rd, "Topic", "/vo_pose");

poses = readMessages(pose_rd);

poses_tbl = struct2table(cell2mat(poses));
poses_tbl.header = struct2table(poses_tbl.header);
poses_tbl.header.stamp = struct2table(poses_tbl.header.stamp);
poses_tbl.pose = struct2table(poses_tbl.pose);
poses_tbl.pose.position = struct2table(poses_tbl.pose.position);
poses_tbl.pose.orientation = struct2table(poses_tbl.pose.orientation);

%%

GT_data = import_data(GT_file);


%%
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

%%
figure;
scatter(poses_tbl.pose.position.x, poses_tbl.pose.position.y)
hold on
scatter(ublox_enu(:,1), ublox_enu(:,2))
axis equal
title("Position SaDVIO")
xlabel("x [m]")
ylabel("y [m]")

% function tbl = expandROS2table(msgs)
%     tbl = struct2table(cell2mat(msgs));
%     for var = string(tbl.Properties.VariableNames)
%         tbl.(var) = iexpandROS2(tbl.(var));
%     end
% 
%     function tbli = iexpandROS2(smsgs)
%         if
%     end
% end