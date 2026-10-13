function [outputArg1,outputArg2] = setup_eval_data(GT_data, pg_pose, lla0)
%SETUP_EVAL_DATA Summary of this function goes here
%   Detailed explanation goes here
arguments (Input)
    inputArg1
    inputArg2
end

arguments (Output)
    outputArg1
    outputArg2
end

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
end