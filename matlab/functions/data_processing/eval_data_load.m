function [  datapath, exp_path, GT_data, lla0, pg_pose, pg_traj, ...
            gnss_est, context_data, context] = eval_data_load( ...
    NVA)
%LOAD_EVAL_DATA Summary of this function goes here
%   Detailed explanation goes here
arguments (Input)
    NVA.exp_path
    NVA.datapath = "D:\m.von-arnim\Documents\data" % Modify as needed
    NVA.load_PG_FUSION  logical     = true
    NVA.load_SADVIO     logical     = true
    NVA.load_ESKF       logical     = true
    NVA.load_UBLOX      logical     = true
    NVA.load_GNSS_EST   logical     = true
    NVA.load_CONTEXT    logical     = true
end

arguments (Output)
    datapath
    exp_path
    GT_data
    lla0
    pg_pose
    pg_traj
    gnss_est
    context_data
    context
end

datapath = NVA.datapath;

%% Select dataset from UI
if ~isfield(NVA,"exp_path") || isempty(NVA.exp_path) || ~isfolder(NVA.exp_path)
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
% rinex_data_folder       = fullfile(datapath, exp_folder, exp_date, "rinex");
context_data_folder     = fullfile(datapath, exp_folder, exp_date, "_results", "context");
vo_data_folder          = fullfile(datapath, exp_folder, exp_date, "_results", "vio");
pg_data_folder          = fullfile(datapath, exp_folder, exp_date, "_results", "pg");
results_est_folder      = fullfile(datapath, exp_folder, exp_date, "_results");

ublox_data_file             = "data_ubx_"+exp_date+".mat";
GT_data_file                = "traj_"+exp_date+".txt";
context_data_file           = "context_pred_"+exp_date+".mat";

pg_bag_folder = uigetdir(pg_data_folder);
gnss_est_folder = uigetdir(results_est_folder);

%% Load GNSS, Context & non-VO data
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

%% Load estimated poses
pg_rd = ros2bagreader(pg_bag_folder);
pg_pose = ros2baggetPoseStamped(select(pg_rd, "Topic", "/pg/enu/pose"));
pgvio_pose = ros2baggetPoseStamped(select(pg_rd, "Topic", "/pg/ecef/pose"));
pg_traj = ros2baggetMarkerPointsFromCSV(pg_bag_folder);
pg_pose.TOW = sync_TOW_rostime(ublox_data.NavPVT.rostime, ublox_data.NavPVT.TOW, pg_pose.rostime_pose);
pg_traj.TOW = sync_TOW_rostime(ublox_data.NavPVT.rostime, ublox_data.NavPVT.TOW, pg_traj.rostime);
pg_traj.NSamp = length(pg_traj.points);

%% Load ESKF relative poses + covariance
vo_rel_pose = ros2baggetPoseWithCovarianceStamped(select(vo_rd, "Topic", "/tf_rel"));
vo_rel_pose.TOW = sync_TOW_rostime(ublox_data.NavPVT.rostime, ublox_data.NavPVT.TOW, vo_rel_pose.rostime_pose);

end