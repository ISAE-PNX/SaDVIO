%% Define data paths
% Modify as needed

% original measurement data root folder
data_dir = fullfile("E:\data\meas_max_cesar\building2\");
% ground truth bag
GT_file = fullfile(data_dir, "traj_building2.txt");
% measurement bags
ublox_bag = fullfile(data_dir, "ublox");
% estimator outputs
vo_bag = fullfile(data_dir, "_res/20260617_162247");
%% Load data
ublox_data = ros2baggetublox(ublox_bag);
vo_rd = ros2bagreader(vo_bag);
%% Load estimated poses
vo_pose = ros2baggetPoseStamped(select(vo_rd, "Topic", "/vo_pose"));
vo_pose.TOW = sync_TOW_rostime(ublox_data.NavPVT.rostime, ublox_data.NavPVT.TOW, vo_pose.rostime_pose);
%% Load ESKF relative poses + covariance
vo_rel_pose = ros2baggetPoseWithCovarianceStamped(select(vo_rd, "Topic", "/tf_rel"));
vo_rel_pose.TOW = sync_TOW_rostime(ublox_data.NavPVT.rostime, ublox_data.NavPVT.TOW, vo_rel_pose.rostime_pose);
%% Load Ground Truth
GT_data = import_data(GT_file);
%% Calculate GT relative poses corresponding to ESKF samples
vars2interp = ["X_ECEF", "Y_ECEF", "Z_ECEF", "Roll", "Pitch", "Heading", "Latitude", "Longitude", "H_Ell"];
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

vo_rel_pose.pos_err = vo_rel_pose.GT.dT.trvec - vo_rel_pose.pos(2:end,:);
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
%% Plot GT & ESKF rel. rotation
t_est = vo_rel_pose.pos(2:end,:);
t_gt = vo_rel_pose.GT.dT.trvec;
figure;
tiledlayout
for ax = 1:3
    nexttile(ax)
    hold on
    plot(t_est(:,ax));
    plot(t_gt(:,ax));
end
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
figure;
tiledlayout
for ax = 1:3
    nexttile(ax)
    hold on
    plot(angles_est(2:end,ax));
    plot(angles_gt(:,ax));
end
%% Plot rel. position error of ESKF
figure;
plot(vo_rel_pose.pos_err(:,1), "DisplayName", "Rel. pose error x")
hold on
plot(sqrt(squeeze(vo_rel_pose.cov(1,1,:))), "b:", "DisplayName", "\sigma")
plot(-sqrt(squeeze(vo_rel_pose.cov(1,1,:))), "b:", "DisplayName", "")
legend
ylabel("Rel. Pose Error [m]")
title("ESKF consistency analysis")
%% Plot rel. position error (Normalized)  of ESKF
figure;
t = vo_rel_pose.GT.dT.trvec;
plot(vo_rel_pose.pos_err(:,1)./t(:,1), "DisplayName", "Rel. pose error x")
hold on
plot(sqrt(squeeze(vo_rel_pose.cov(1,1,2:end)))./t(:,1), "b:", "DisplayName", "\sigma")
plot(-sqrt(squeeze(vo_rel_pose.cov(1,1,2:end)))./t(:,1), "b:", "DisplayName", "")
legend
ylabel("Rel. Pose Error Normalized [%]")
title("ESKF consistency analysis")
ylim([-5 5])
%% Plot rel. rotation error of ESKF
figure;
plot(vo_rel_pose.rot_err)
hold on
for i = 1:vo_rel_pose.NSamp
    c(i) = trace(vo_rel_pose.cov(4:6,4:6,i));
end
plot(sqrt(c), "b:")
% plot(sqrt(squeeze(vo_rel_pose.cov(5,5,:))), "r:")
% plot(sqrt(squeeze(vo_rel_pose.cov(4,4,:))), "g:")
%% Histograms ...
figure
histogram(vo_rel_pose.pos_err(:,1))
%% NEES
figure
hold on
scatter(1:vo_rel_pose.NSamp, vo_rel_pose.pos_nees, ".")
hline(lowerlimit);
hline(upperlimit);
%% NEES histogram
figure
hold on
histogram(vo_rel_pose.pos_nees, 1000)
% quantile(vo_rel_pose.pos_nees, 1, )
vline(lowerlimit)
vline(upperlimit)