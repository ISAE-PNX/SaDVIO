init_path
%% Define data paths
% Modify as needed
datapath = "D:\m.von-arnim\Documents\data";
if ~exist("exp_path", "var")
    % [  datapath, exp_path, GT_data, pg_pose, pg_traj, ...
    %             gnss_est, context_data, context] = eval_data_load(...
    %         exp_path="",
else
    [  datapath, exp_path, GT_data, lla0, pg_pose, pg_traj, ...
                gnss_est, context_data, context] = eval_data_load(...
            exp_path=exp_path, ...
            datapath=datapath, ...
            load_PG_FUSION= true, ...
            load_GNSS_EST=  true, ...
            load_UBLOX=     true, ...
            load_CONTEXT=   true, ...
            load_ESKF=      false, ...
            load_SADVIO=    false);
end

%% output path
% if this does not work, run LOAD_DATA
[resultpath, imgfolder] = setup_output(exp_date, "ublox");
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