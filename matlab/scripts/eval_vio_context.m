init_path
%% Define data paths
% Modify as needed

% original measurement data root folder
% data_dir = fullfile("E:\data\meas_max_cesar\building2\");
data_dir = fullfile("E:/data/sky_stereo_gnss_lidar_laas/");
res_dir = fullfile("I:/max/no_context_gvio_sadvio_results/sky_stereo_gnss_lidar_laas");
% ground truth bag
% GT_file = fullfile(data_dir, "traj_building2.txt");
GT_file = fullfile(data_dir, "traj_novatel/traj_2025-11-04-13-34-21.txt");
% measurement bags
ublox_bag = fullfile(data_dir, "ublox");
% estimator outputs
% vo_bag = fullfile(data_dir, "_res/20260617_162247");
vo_bag = fullfile(res_dir, "_res/vio/20260702_182156/");
% context

exp_date = "2025-11-04-13-34-21";
context_data_folder  = fullfile(data_dir, "_results", "context");
context_data_file    = "context_pred_"+exp_date+".mat";

%% Load data
ublox_data = ros2baggetublox(ublox_bag);
vo_rd = ros2bagreader(vo_bag);

[context_data, context]         = supaero_loadEnvContext( ...
    "folder", context_data_folder,      "file", context_data_file, ...
    "WeekNum", gnssRawUblox.WeekNum(1), ...
    "rostimeref", gnssNavPvtUblox);
%% Load estimated poses
vo_pose = ros2baggetPoseStamped(select(vo_rd, "Topic", "/vo_pose"));
vo_pose.TOW = sync_TOW_rostime(ublox_data.NavPVT.rostime, ublox_data.NavPVT.TOW, vo_pose.rostime_pose);
%% Load ESKF relative poses + covariance
vo_rel_pose = ros2baggetPoseWithCovarianceStamped(select(vo_rd, "Topic", "/tf_rel"));
vo_rel_pose.TOW = sync_TOW_rostime(ublox_data.NavPVT.rostime, ublox_data.NavPVT.TOW, vo_rel_pose.rostime_pose);
%% Load Ground Truth
GT_data = import_data(GT_file);
%% Calculate GT relative poses corresponding to ESKF samples
vars2interp = ["X_ECEF", "Y_ECEF", "Z_ECEF", "Roll", "Pitch", "Heading", "Latitude", "Longitude", "H_Ell", ...
    "VEast", "VNorth", "VUp"];
for v = vars2interp
    vo_rel_pose.GT.(v) = interp1(GT_data.GPSTime, GT_data.(v), vo_rel_pose.TOW);
end
vo_rel_pose.GT.ENU = lla2enu([vo_rel_pose.GT.Latitude, vo_rel_pose.GT.Longitude, vo_rel_pose.GT.H_Ell], ...
    [vo_rel_pose.GT.Latitude(1), vo_rel_pose.GT.Longitude(1), vo_rel_pose.GT.H_Ell(1)],"flat");
vo_rel_pose.GT.NED = lla2ned([vo_rel_pose.GT.Latitude, vo_rel_pose.GT.Longitude, vo_rel_pose.GT.H_Ell], ...
    [vo_rel_pose.GT.Latitude(1), vo_rel_pose.GT.Longitude(1), vo_rel_pose.GT.H_Ell(1)],"flat");
%% Set up GT pose
vo_rel_pose.GT.T(vo_rel_pose.NSamp) = se3;
for i = 1:vo_rel_pose.NSamp
    vo_rel_pose.GT.T(i) = se3([vo_rel_pose.GT.Roll(i), vo_rel_pose.GT.Pitch(i), vo_rel_pose.GT.Heading(i)], "eul", "xyz",...
        vo_rel_pose.GT.NED(i,:));
end
%% Set up VO pose
vo_pose.T(vo_pose.NSamp) = se3;
for i = 1:vo_pose.NSamp
    vo_pose.T(i) = se3(vo_pose.rot(i), vo_pose.pos(i,:));
end
%% Extract position error
% vo_rel_pose.GT.dT(vo_rel_pose.NSamp-1) = se3;
% vo_rel_pose.GT.dT = vo_rel_pose.GT.dT';
for i = 1:vo_rel_pose.NSamp-1
    vo_rel_pose.GT.dT(i) = vo_rel_pose.GT.T(i).inv * vo_rel_pose.GT.T(i+1);
end
vo_rel_pose.GT.dT = vo_rel_pose.GT.dT';

% vo_rel_pose.pos_err = vo_rel_pose.GT.dT.trvec - vo_rel_pose.pos(2:end,:);
vo_rel_pose.pos_err = vo_rel_pose.GT.dT.trvec - vo_rel_pose.pos(2:end,:) / sscale;
%% Extract rotation error
enu2ned = so3([0 1 0; 0 1 0; 0 0 -1]);
vo_rel_pose.rot_err = vo_rel_pose.GT.dT.so3.quaternion.dist(enu2ned.quaternion * vo_rel_pose.rot(2:end));
ang_est = vo_rel_pose.rot.euler("ZYX","frame");
ang_est(:,1) = ang_est(:,1);
vo_rel_pose.rot_err_eul = vo_rel_pose.GT.dT.eul - ang_est(2:end,:);

%% normalized rel. pose error
vo_rel_pose.pos_err_n = vo_rel_pose.pos_err ./ vo_rel_pose.GT.dT.trvec;
%% assign context
dtm = vo_rel_pose.datetime;
dtm.TimeZone = "UTCLeapSeconds";
vo_rel_pose.context = interp1( ...
            context_data.datetime, context_data.context, ...
            dtm, ... % interpolate measurements from meas_TOW to rx_TOW
            "nearest")';
%% plot relative error statistics by context
plot_ecdf_context(figure(), repmat(vo_rel_pose.pos_err(:,1)', 3,1), ...
    vo_rel_pose.context(2:end), context.list, ...
    "DisplayName", "vo", "Color", "b");

linkaxes(findall(gcf(),'type','axes'), 'xy')
xlabel("RElative displacement error [-]")
xticks(0:0.1:1)
xlim([0 1])
title(gcf().Children, "Relative Translation Error Normalized")
legend
%% Again, but excluding stationary points
% venu = [vo_rel_pose.GT.VEast, vo_rel_pose.GT.VNorth, vo_rel_pose.GT.VUp];
% stationary_sel = vecnorm(venu,2,2) < 0.01;
% context_selected = vo_rel_pose.context(stationary_sel);

plot_ecdf_context(figure(), ...
    100*[vo_rel_pose.pos_err(~stationary_sel(2:end),1)'; zeros(2, length(vo_rel_pose.pos_err(~stationary_sel(2:end),1)'))], ...
    vo_rel_pose.context(~stationary_sel), context.list, ...
    "DisplayName", "vo", "Color", "b");

linkaxes(findall(gcf(),'type','axes'), 'xy')
xlabel("Displacement error [cm]")
xticks(0:0.1:1)
xlim([0 5])
xticks auto
title(gcf().Children, "Relative Translation Error in Forward Direction", "per KeyFrame pair")
legend
%% plot normlized relative error statistics by context
plot_ecdf_context(figure(), 100*[vo_rel_pose.pos_err_n(:,1)'; zeros(2, length(vo_rel_pose.pos_err_n(~stationary_sel(2:end),1)'))], ...
    vo_rel_pose.context(2:end), context.list, ...
    "DisplayName", "vo", "Color", "b");

linkaxes(findall(gcf(),'type','axes'), 'xy')
xlabel("RElative displacement error [\%]")
xticks(0:0.1:1)
xlim([0 1])
title(gcf().Children, "Relative Translation Error in Forward Direction, Normalized", "per KeyFrame pair")
legend
%% Again, but excluding stationary points
% venu = [vo_rel_pose.GT.VEast, vo_rel_pose.GT.VNorth, vo_rel_pose.GT.VUp];
% stationary_sel = vecnorm(venu,2,2) < 0.01;
% context_selected = vo_rel_pose.context(stationary_sel);

plot_ecdf_context(figure(), ...
    100*[vo_rel_pose.pos_err_n(~stationary_sel(2:end),1)'; zeros(2, length(vo_rel_pose.pos_err_n(~stationary_sel(2:end),1)'))], ...
    vo_rel_pose.context(~stationary_sel), context.list, ...
    "DisplayName", "vo", "Color", "b");

linkaxes(findall(gcf(),'type','axes'), 'xy')
xlabel("Relative displacement error [%]")
xticks(0:0.1:1)
xlim([0 100])
for ax = findall(gcf(),'type','axes')
    xticks(ax, 'auto')
end
title(gcf().Children, "Relative Translation Error in Forward Direction, Normalized", "per KeyFrame pair")
legend
%% plot relative displacement per keyframe statistics by context
plot_ecdf_context(figure(), repmat(vo_rel_pose.pos(:,1)', 3,1), ...
    vo_rel_pose.context, context.list, ...
    "DisplayName", "vo", "Color", "b");

linkaxes(findall(gcf(),'type','axes'), 'xy')
xlabel("Displacement [m]")
xticks(0:0.1:1)
xlim([0 0.5])
title(gcf().Children, "Forward VO displacement per KeyFrame")
legend
%% Again, but excluding stationary points
plot_ecdf_context(figure(), (100 / sscale)*[vo_rel_pose.pos(~stationary_sel,1)'; zeros(2, length(vo_rel_pose.pos(~stationary_sel,1)'))], ...
    vo_rel_pose.context(~stationary_sel), context.list, ...
    "DisplayName", "vo", "Color", "b");

linkaxes(findall(gcf(),'type','axes'), 'xy')
xlabel("Displacement [cm]")
xlim([0 30])
for ax = findall(gcf(),'type','axes')
    xticks(ax, 0:5:30)
end
title(gcf().Children, "Forward VO displacement per KeyFrame")
legend
%% plot relative displacement covariance statistics by context
ct = zeros(vo_rel_pose.NSamp,1);
for i = 1:vo_rel_pose.NSamp
    ct(i) = sqrt(trace(vo_rel_pose.cov(4:6,4:6,i)));
end
ctn = ct(2:end) ./ vecnorm(vo_rel_pose.GT.dT.trvec,2,2);

plot_ecdf_context(figure(), [ctn'; zeros(2, length(ctn'))], ...
    vo_rel_pose.context, context.list, ...
    "DisplayName", "vo", "Color", "b");

linkaxes(findall(gcf(),'type','axes'), 'xy')
xlabel("Displacement [m]")
xlim([0 0.5])
for ax = findall(gcf(),'type','axes')
    xticks(0:0.1:1)
end
title(gcf().Children, "ESKF Covariance per KeyFrame")
legend
%% Again, but excluding stationary points
plot_ecdf_context(figure(), repmat(vo_rel_pose.pos(~stationary_sel,1)', 3,1), ...
    vo_rel_pose.context(~stationary_sel), context.list, ...
    "DisplayName", "vo", "Color", "b");

linkaxes(findall(gcf(),'type','axes'), 'xy')
xlabel("Displacement [m]")
xticks(0:0.1:1)
xlim([0 0.5])
title(gcf().Children, "Forward VO displacement per KeyFrame")
legend