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
vo_data_folder          = fullfile(datapath, exp_folder, exp_date, "_results", "vio");
% pg_data_folder          = fullfile(datapath, exp_folder, exp_date, "_results", "pg");

ublox_data_file             = "data_ubx_"+exp_date+".mat";
GT_data_file                = "traj_"+exp_date+".txt";
context_data_file           = "context_pred_"+exp_date+".mat";

vo_bag_folder = uigetdir(vo_data_folder);
%% Load data
% try
    % [ublox_data.gnssRaw, gnssNavSatUblox, ublox_data.NavPVT] = supaero_loadUbloxGNSS( ...
    %     "folder", ublox_data_folder,        "file", ublox_data_file);
    ublox_data = ros2baggetublox(ublox_data_folder);
    ublox_data.enu = lla2enu([ublox_data.NavSatFix.latitude, ublox_data.NavSatFix.longitude, ublox_data.NavSatFix.altitude], ...
        [ublox_data.NavSatFix.latitude(1), ublox_data.NavSatFix.longitude(1), ublox_data.NavSatFix.altitude(1)], "flat");

% catch e
%     disp(e)
% end
% try
[GT_data]                       = supaero_loadGT( ...
    "folder", GT_data_folder,           "file", GT_data_file, ...
    "WeekNum", ublox_data.RAWX.WeekNum(1));
% catch e
%     disp(e)
% end
[context_data, context]         = supaero_loadEnvContext( ...
    "folder", context_data_folder,      "file", context_data_file, ...
    "WeekNum", ublox_data.RAWX.WeekNum(1), ...
    "rostimeref", ublox_data.NavPVT);
%% Load estimated poses
vo_rd = ros2bagreader(vo_bag_folder);
vo_pose = ros2baggetPoseStamped(select(vo_rd, "Topic", "/vo_pose"));
vo_pose.TOW = sync_TOW_rostime(ublox_data.NavPVT.rostime, ublox_data.NavPVT.TOW, vo_pose.rostime_pose);
%% output path
% if this does not work, run LOAD_DATA
if ~exist("resultfile", "var")
    % resultfile = sprintf("%s_%s_res.mat", date, gnssRaw.rec);
    resultfile = sprintf("%s_%s_res_2.mat", exp_date, "ublox");
end
resultfolder =  fullfile(vo_bag_folder);
imgfolder = fullfile(resultfolder, "fig");
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
%% Load ESKF relative poses + covariance
vo_rel_pose = ros2baggetPoseWithCovarianceStamped(select(vo_rd, "Topic", "/tf_rel"));
vo_rel_pose.TOW = sync_TOW_rostime(ublox_data.NavPVT.rostime, ublox_data.NavPVT.TOW, vo_rel_pose.rostime_pose);
%% Calculate GT relative poses corresponding to ESKF samples
vars2interp = ["X_ECEF", "Y_ECEF", "Z_ECEF", "Roll", "Pitch", "Heading", "Latitude", "Longitude", "H_Ell", ...
    "VEast", "VNorth", "VUp"];
for v = vars2interp
    vo_rel_pose.GT.(v) = interp1(GT_data.GPSTime, GT_data.(v), vo_rel_pose.TOW);
end

vo_rel_pose.GT.vENU = [vo_rel_pose.GT.VEast, vo_rel_pose.GT.VNorth, vo_rel_pose.GT.VUp];
stationary_v_m_per_s = 0.05;
stationary_sel = vecnorm(vo_rel_pose.GT.vENU,2,2) < stationary_v_m_per_s;

vo_rel_pose.GT.ENU = lla2enu([vo_rel_pose.GT.Latitude, vo_rel_pose.GT.Longitude, vo_rel_pose.GT.H_Ell], ...
    [vo_rel_pose.GT.Latitude(1), vo_rel_pose.GT.Longitude(1), vo_rel_pose.GT.H_Ell(1)],"flat");
vo_rel_pose.GT.NED = lla2ned([vo_rel_pose.GT.Latitude, vo_rel_pose.GT.Longitude, vo_rel_pose.GT.H_Ell], ...
    [vo_rel_pose.GT.Latitude(1), vo_rel_pose.GT.Longitude(1), vo_rel_pose.GT.H_Ell(1)],"flat");
%% Set up GT pose
vo_rel_pose.GT.T = se3([vo_rel_pose.GT.Roll, vo_rel_pose.GT.Pitch, vo_rel_pose.GT.Heading], "eul", "xyz",...
        vo_rel_pose.GT.NED);
%% Set up VO pose
vo_pose.T = se3(vo_pose.rot, vo_pose.pos);
%% Extract relative position error
vo_rel_pose.GT.dT = vo_rel_pose.GT.T(1:end-1).inv .* vo_rel_pose.GT.T(2:end);
vo_rel_pose.GT.dT = vo_rel_pose.GT.dT';

% vo_rel_pose.pos_err = vo_rel_pose.GT.dT.trvec - vo_rel_pose.pos(2:end,:);
vo_rel_pose.pos_err = vo_rel_pose.GT.dT.trvec - vo_rel_pose.pos(2:end,:) / sscale;
%% Extract rotation error
enu2ned = so3([0 1 0; 0 1 0; 0 0 -1]);
vo_rel_pose.rot_err = vo_rel_pose.GT.dT.so3.quaternion.dist(enu2ned.quaternion * vo_rel_pose.rot(2:end));
ang_est = vo_rel_pose.rot.euler("ZYX","frame");
ang_est(:,1) = ang_est(:,1);
vo_rel_pose.rot_err_eul = vo_rel_pose.GT.dT.eul - ang_est(2:end,:);

%% Calculate rel. position NEES
vo_rel_pose.pos_nees = zeros(vo_rel_pose.NSamp,1);
for i = 1:vo_rel_pose.NSamp-1
    vo_rel_pose.pos_nees(i) = vo_rel_pose.pos_err(i,:) * (vo_rel_pose.cov(1:3,1:3,i) \ vo_rel_pose.pos_err(i,:)');
end
confidence = 0.05;
lowerlimit = icdf("Chisquare",confidence/2,3);
upperlimit = icdf("Chisquare",1-confidence/2,3);
%% Calculate rel. position ANEES
anees = mean(vo_rel_pose.pos_nees);
lowerlimit_anees = icdf("Chisquare",confidence/2,3*vo_rel_pose.NSamp);
upperlimit_anees = icdf("Chisquare",confidence/2,3*vo_rel_pose.NSamp);
%% %%%% FIGURES and PLOTS %%%% %%
%% Plot GT pose
figure
hold on
for i = 1:10:vo_rel_pose.NSamp
    % plotframe(vo_rel_pose.GT.T(i).R, vo_rel_pose.GT.T(i).Translation)
    plotTransforms(vo_rel_pose.GT.T(i), "InertialZDirection", "Down")
end
xlabel("N")
ylabel("E")
zlabel("U")
axis equal
%% Plot VO pose
figure
hold on
for i = 1:10:vo_pose.NSamp
    % plotframe(vo_rel_pose.GT.T(i).R, vo_rel_pose.GT.T(i).Translation)
    plotTransforms(vo_pose.T(i), "InertialZDirection", "Up")
end
xlabel("N")
ylabel("E")
zlabel("U")
axis equal
%% Plot the VIO position
figure;
scatter(vo_pose.pos(:,1), vo_pose.pos(:,2))
axis equal
title("Position SaDVIO")
xlabel("x [m]")
ylabel("y [m]")
%% Plot the GT position axes
figure
plot(diff(vecnorm(vo_rel_pose.GT.T.trvec,2,2)))
%% Plot the GT horizontal position
figure
t = vo_rel_pose.GT.T.trvec;
plot(t(:,2), t(:,1))
xlabel("E")
ylabel("N")
%% Plot the VO position axes
figure
plot(diff(vecnorm(vo_pose.T.trvec,2,2)))
%% Plot GT & ESKF rel. position
t_est = vo_rel_pose.pos(2:end,:) / sscale;
t_gt = vo_rel_pose.GT.dT.trvec;
freleskf = figure;
tiledlayout
axis_order = ["x", "y", "z"];
for ax = 1:3
    nexttile(ax)
    hold on
    plot(t_est(:,ax), "DisplayName", "Estimate");
    plot(t_gt(:,ax), "DisplayName", "Ground truth");
    title(axis_order(ax))
    ylabel("Displacement [m]")
    xlabel("Frame Index")
end
l = legend;
linkaxes(findall(freleskf,'type','axes'), 'xy')
%% Plot the GT rel. pose
figure;
% t = zeros(vo_rel_pose.NSamp-1,3);
% for i = 1:vo_rel_pose.NSamp-1
%     t(i,:) = vo_rel_pose.GT.dT(i).trvec;
% end
% plot(t)
% plot(vo_rel_pose.GT.dT.trvec)
% legend("x","y","z")
plot(vo_rel_pose.GT.dT.eul)
legend("Yaw","Pitch","Roll")
%% Plot the ESKF rel. pose
figure;
% plot(vo_rel_pose.pos)
% legend("x","y","z")
angles = vo_rel_pose.rot.euler("ZYX", "frame");
angles(:,1) = -angles(:,1);
plot(angles)
legend("Yaw","Pitch","Roll")
%% Plot GT & ESKF rel. rotation
angles_est = vo_rel_pose.rot.euler("ZYX", "frame");
angles_est(:,1) = -angles_est(:,1);
angles_gt = vo_rel_pose.GT.dT.eul;
froteskf = figure;
tiledlayout
axis_order = ["pitch", "roll", "yaw"];
for ax = 1:3
    nexttile(ax)
    hold on
    plot(angles_est(2:end,ax));
    plot(angles_gt(:,ax));
    title(axis_order(ax))
    ylabel("Rotation [rad]")
    xlabel("Frame Index")
end
linkaxes(findall(froteskf,'type','axes'), 'xy')
%% Plot rel. position error of ESKF
figure;
colororder("sail")
hold on
p1 = plot(sqrt(squeeze(vo_rel_pose.cov(1,1,:))), ":", "DisplayName", "ESKF estimated std. dev. of RTE");
p2 = plot(-sqrt(squeeze(vo_rel_pose.cov(1,1,:))), ":", "DisplayName", "-\sigma", "HandleVisibility", "off");
p2.Color = p1.Color;
plot(vo_rel_pose.pos_err(:,1), "DisplayName", "Along-track VO RTE")
legend
ylabel("Relative Translation Error [m]")
xlabel("Keyframe index")
title("ESKF consistency analysis")
%% Plot rel. position error (Normalized)  of ESKF
figure;
t = vo_rel_pose.GT.dT.trvec;
plot(vo_rel_pose.pos_err(~stationary_sel(2:end),1)./t(~stationary_sel(2:end),1), "DisplayName", "Rel. pose error x")
hold on
% plot(sqrt(squeeze(vo_rel_pose.cov(1,1,~stationary_sel(2:end))))./t(~stationary_sel(2:end),1), "b:", "DisplayName", "\sigma")
% plot(-sqrt(squeeze(vo_rel_pose.cov(1,1,~stationary_sel(2:end))))./t(~stationary_sel(2:end),1), "b:", "DisplayName", "-\sigma")
legend
ylabel("Rel. Pose Error Normalized [%]")
title("ESKF consistency analysis")
ylim([-1 1])
%% Plot rel. rotation error of ESKF
figure;
plot(pi-vo_rel_pose.rot_err)
hold on
for i = 1:vo_rel_pose.NSamp
    c(i) = trace(vo_rel_pose.cov(1:3,1:3,i));
end
plot(sqrt(c), "b:")
% plot(sqrt(squeeze(vo_rel_pose.cov(5,5,:))), "r:")
% plot(sqrt(squeeze(vo_rel_pose.cov(4,4,:))), "g:")
%% Histograms ...
figure
% histogram(vo_rel_pose.pos_err(:,1), Normalization="percentage")
% xlim([-0.05 0.05])
histogram(vo_rel_pose.pos_err_n(~stationary_sel,1), ...
    min(vo_rel_pose.pos_err_n(~stationary_sel,1)):0.01:max(vo_rel_pose.pos_err_n(~stationary_sel,1)))
xlim([-0.3 0.3])
title("Along-track error of relative pose estimate")
ylabel("[%]")
xlabel("Error [m]")

%% NEES
figure
hold on
scatter(1:vo_rel_pose.NSamp, vo_rel_pose.pos_nees, ".")
hline(lowerlimit);
hline(upperlimit);
%% NEES histogram
figure
hold on
histogram(vo_rel_pose.pos_nees, 0:0.1:max(vo_rel_pose.pos_nees))
% quantile(vo_rel_pose.pos_nees, 1, )
vline(lowerlimit)
vline(upperlimit)
xlim([0 2*upperlimit])