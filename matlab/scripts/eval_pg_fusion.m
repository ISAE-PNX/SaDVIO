init_path
%% Define data paths
% Modify as needed
datapath = "D:\m.von-arnim\Documents\data";

% use UI instead
if ~exist("exp_path", "var")
    exp_path = uigetdir(datapath);
else
    if questdlg("Select new data?") == "Yes"   
        exp_path = uigetdir(datapath);
    end
end
disp(exp_path) % what the user selected

exp_path = split(string(exp_path), filesep);
datapath = strjoin(exp_path(1:end-2), filesep);
exp_date = exp_path(end);
exp_folder = exp_path(end-1);

disp(fullfile(datapath, exp_folder, exp_date))

GT_data_folder          = fullfile(datapath, exp_folder, exp_date);
ublox_data_folder       = fullfile(datapath, exp_folder, exp_date, "ublox");
rinex_data_folder       = fullfile(datapath, exp_folder, exp_date, "rinex");
context_data_folder     = fullfile(datapath, exp_folder, exp_date, "_results", "context");
pg_data_folder          = fullfile(datapath, exp_folder, exp_date, "_results", "pg");
% pg_data_folder          = fullfile(datapath, exp_folder, exp_date, "_results", "pg");
results_est_folder      = fullfile(datapath, exp_folder, exp_date, "_results");%, "WLSQ__ms_A___GPS_GAL__context");
% gnss_est_folder         = fullfile(datapath, exp_folder, exp_date, "_results", "WLSQ__ms___GPS_GAL__context");

ublox_data_file             = "data_ubx_"+exp_date+".mat";
GT_data_file                = "traj_"+exp_date+".txt";
context_data_file           = "context_pred_"+exp_date+".mat";

pg_bag_folder = uigetdir(pg_data_folder);
gnss_est_folder = uigetdir(results_est_folder);
%% Load data
% try
    % [ublox_data.gnssRaw, gnssNavSatUblox, ublox_data.NavPVT] = supaero_loadUbloxGNSS( ...
    %     "folder", ublox_data_folder,        "file", ublox_data_file);
ublox_data = ros2baggetublox(ublox_data_folder);
lla0 = [ublox_data.NavSatFix.latitude(1), ublox_data.NavSatFix.longitude(1), ublox_data.NavSatFix.altitude(1)];
ublox_data.enu = lla2enu([ublox_data.NavSatFix.latitude, ublox_data.NavSatFix.longitude, ublox_data.NavSatFix.altitude], ...
    lla0, "flat");

[GT_data]                       = supaero_loadGT( ...
    "folder", GT_data_folder,           "file", GT_data_file, ...
    "WeekNum", ublox_data.RAWX.WeekNum(1));
[context_data, context]         = supaero_loadEnvContext( ...
    "folder", context_data_folder,      "file", context_data_file, ...
    "WeekNum", ublox_data.RAWX.WeekNum(1), ...
    "rostimeref", ublox_data.NavPVT);

gnss_est = ros2baggetNavSatFixContext(gnss_est_folder);    
gnss_est.enu = lla2enu([gnss_est.latitude, gnss_est.longitude, gnss_est.altitude], ...
        lla0, "flat");

%% output path
% if this does not work, run LOAD_DATA
if ~exist("resultfile", "var")
    % resultfile = sprintf("%s_%s_res.mat", date, gnssRaw.rec);
    resultfile = sprintf("%s_%s_res_2.mat", exp_date, "ublox");
end
resultfolder =  fullfile(pg_bag_folder);
imgfolder = fullfile(resultfolder, "fig");
if ~isfolder(imgfolder)
    mkdir(imgfolder);
end
resultpath = fullfile(resultfolder, resultfile);
if ~isfolder(resultfolder)
    mkdir(resultfolder);
end
if ~isfile(resultpath)
    save(resultpath, 'exp_date', "-v7.3");
end
% check what is already in the results file
disp(resultfile)
who('-file', resultpath);
%% Load estimated poses
pg_rd = ros2bagreader(pg_bag_folder);
pg_pose = ros2baggetPoseStamped(select(pg_rd, "Topic", "/pg/enu/pose"));
pgvio_pose = ros2baggetPoseStamped(select(pg_rd, "Topic", "/pg/ecef/pose"));
pg_traj = ros2baggetMarkerPointsFromCSV(pg_bag_folder);
pg_pose.TOW = sync_TOW_rostime(ublox_data.NavPVT.rostime, ublox_data.NavPVT.TOW, pg_pose.rostime_pose);
pg_traj.TOW = sync_TOW_rostime(ublox_data.NavPVT.rostime, ublox_data.NavPVT.TOW, pg_traj.rostime);
pg_traj.NSamp = length(pg_traj.points);
%% Calculate GT relative poses corresponding to ESKF samples
vars2interp = ["X_ECEF", "Y_ECEF", "Z_ECEF", "Roll", "Pitch", "Heading", "Latitude", "Longitude", "H_Ell"];
for v = vars2interp
    pg_pose.GT.(v) = interp1(GT_data.GPSTime, GT_data.(v), pg_pose.TOW);
    % pg_traj.GT.(v) = interp1(GT_data.GPSTime, GT_data.(v), pg_traj.TOW);
end
pg_pose.GT.ENU = lla2enu([pg_pose.GT.Latitude, pg_pose.GT.Longitude, pg_pose.GT.H_Ell], ...
    lla0,"flat");
pg_pose.GT.NED = lla2ned([pg_pose.GT.Latitude, pg_pose.GT.Longitude, pg_pose.GT.H_Ell], ...
    lla0,"flat");
% pg_traj.GT.ENU = lla2enu([pg_traj.GT.Latitude, pg_traj.GT.Longitude, pg_traj.GT.H_Ell], ...
%     lla0,"flat");
% pg_traj.GT.NED = lla2ned([pg_traj.GT.Latitude, pg_traj.GT.Longitude, pg_traj.GT.H_Ell], ...
%     lla0,"flat");
%% Set up GT pose
pg_pose.GT.T(pg_pose.NSamp) = se3;
pg_pose.GT.T = se3([pg_pose.GT.Roll, pg_pose.GT.Pitch, pg_pose.GT.Heading], "eul", "xyz",...
    pg_pose.GT.NED);
% pg_traj.GT.T() = se3;
% pg_traj.GT.T = se3([pg_traj.GT.Roll, pg_traj.GT.Pitch, pg_traj.GT.Heading], "eul", "xyz",...
%     pg_traj.GT.NED);
%% Set up VO pose
pg_pose.T(pg_pose.NSamp) = se3;
pg_pose.T = se3(pg_pose.rot, pg_pose.pos);
%% Extract position error
pg_pose.pos_err = pg_pose.GT.T.trvec - pg_pose.pos;
% pg_traj.pos_err = pg_pose.GT.T.trvec - pg_traj.points;
%% Extract rotation error
enu2ned = so3([0 1 0; 0 1 0; 0 0 -1]);
pg_pose.rot_err = pg_pose.GT.T.so3.quaternion.dist(enu2ned.quaternion * pg_pose.rot');
ang_est = pg_pose.rot.euler("ZYX","frame");
ang_est(:,1) = ang_est(:,1);
pg_pose.rot_err_eul = pg_pose.GT.T.eul - ang_est;
%% get GNSS used flag
cov_diag = zeros(3, size(gnss_est.position_covariance,3));
for i = 1:size(gnss_est.position_covariance,3)
    cov_diag(:,i) = diag(gnss_est.position_covariance(:,:,i));
end
threshold = 20;
gns_est.is_used = vecnorm(cov_diag,2,1) < threshold;
%% Calculate rel. position NEES
pg_rel_pose.pos_nees = zeros(pg_rel_pose.NSamp,1);
for i = 1:pg_rel_pose.NSamp-1
    pg_rel_pose.pos_nees(i) = pg_rel_pose.pos_err(i,:) * (pg_rel_pose.cov(1:3,1:3,i) \ pg_rel_pose.pos_err(i,:)');
end
confidence = 0.05;
lowerlimit = icdf("Chisquare",confidence/2,3);
upperlimit = icdf("Chisquare",1-confidence/2,3);
%% Calculate rel. position ANEES
anees = mean(pg_rel_pose.pos_nees);
lowerlimit_anees = icdf("Chisquare",confidence/2,3*pg_rel_pose.NSamp);
upperlimit_anees = icdf("Chisquare",confidence/2,3*pg_rel_pose.NSamp);
%% %%%% FIGURES and PLOTS %%%% %%
%% Plot GT pose
figure
hold on
for i = 1:1000:pg_pose.NSamp
    % plotframe(vo_rel_pose.GT.T(i).R, vo_rel_pose.GT.T(i).Translation)
    plotTransforms(pg_pose.GT.T(i), "InertialZDirection", "Down")
end
xlabel("N")
ylabel("E")
zlabel("U")
%% Plot VO pose
figure
hold on
for i = 1:1000:pg_pose.NSamp
    % plotframe(vo_rel_pose.GT.T(i).R, vo_rel_pose.GT.T(i).Translation)
    plotTransforms(pg_pose.T(i), "InertialZDirection", "Up")
end
xlabel("N")
ylabel("E")
zlabel("U")
%% Plot the VIO position
figure;
hold on

scatter(gnss_est.enu(gns_est.is_used,1), gnss_est.enu(gns_est.is_used,2), 'g', ...
    'Marker', '.', ...
    'DisplayName', "wlsq")
scatter(gnss_est.enu(~gns_est.is_used,1), gnss_est.enu(~gns_est.is_used,2), 'r', ...
    'Marker', '.', ...
    'DisplayName', "wlsq - not used")

% scatter(pg_pose.pos(:,1), pg_pose.pos(:,2))
plot(pg_pose.pos(:,1), pg_pose.pos(:,2), 'c', 'DisplayName', "PG-Fusion (real-time)")
plot(pg_traj.points(1,:), pg_traj.points(2,:), 'b', 'DisplayName', "PG-Fusion (total trajectory)")
plot(pg_pose.GT.ENU(:,1), pg_pose.GT.ENU(:,2), 'k', 'DisplayName', "Ground truth")
% plot(pg_pose.GT.ENU(:,1)+en_offset(1), pg_pose.GT.ENU(:,2)+en_offset(2), 'k')
plot(ublox_data.enu(:,1), ublox_data.enu(:,2), 'k--', 'DisplayName', "ublox")
% plot(gnss_est.enu(:,1), gnss_est.enu(:,2), 'r', 'DisplayName', "wlsq")

axis equal
title("Position SaDVIO")
xlabel("E [m]")
ylabel("N [m]")
l = legend("Location", "northoutside");
% l.Tile = "NorthOutside"
rectangle("Position", [-346 136 20 16], "LineWidth", 1)
text(-320, 140, "A", "FontSize", 14)
rectangle("Position", [-270 -35 60 40], "LineWidth", 1)
text(-205, 0, "B", "FontSize", 14)
rectangle("Position", [-35 -100 18 12], "LineWidth", 1)
text(-35, -75, "C", "FontSize", 14)
% rectangle("Position", [-150 -120 100 40], "LineWidth", 1)
% text(-150, -60, "D", "FontSize", 14)
%% Plot rotation error of PG FUSION
figure;
plot(pg_pose.datetime, rad2deg(pi-pg_pose.rot_err))
ylabel("Angular error [deg]")
xlabel("time")
%% align horizontally
GT_data.datetime = gps_2_utc_vec(GT_data.GPSTime, ublox_data.RAWX.WeekNum(1));
vars2interp = ["X_ECEF", "Y_ECEF", "Z_ECEF", "Roll", "Pitch", "Heading", "Latitude", "Longitude", "H_Ell"];
for v = vars2interp
    ublox_data.GT.(v) = interp1(GT_data.datetime, GT_data.(v), gps_2_utc_vec(ublox_data.RAWX.TOW, ublox_data.RAWX.WeekNum(1)));
end
ublox_data.GT.ENU = lla2enu([ublox_data.GT.Latitude, ublox_data.GT.Longitude, ublox_data.GT.H_Ell], ...
    lla0,"flat");
sel = isfinite(ublox_data.GT.ENU(:,1));
en_offset = median(ublox_data.enu(sel,1:2) - ublox_data.GT.ENU(sel,1:2), 1)