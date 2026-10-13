function [resultdatapath, resultimgpath] = setup_output(exp_date, abs_pose_provider)
%SETUP_OUTPUT Summary of this function goes here
%   Detailed explanation goes here
% if this does not work, run LOAD_DATA
arguments (Input)
    exp_date
    abs_pose_provider = "ublox"
end

arguments (Output)
    resultdatapath
end

% if this does not work, run LOAD_DATA
if ~exist("resultfile", "var")
    resultfile = sprintf("%s_%s_res_2.mat", exp_date, abs_pose_provider);
end
resultfolder =  fullfile(pg_bag_folder);

resultimgpath = fullfile(resultfolder, "fig");
if ~isfolder(resultimgpath)
    mkdir(resultimgpath);
end

resultdatapath = fullfile(resultfolder, resultfile);
if ~isfolder(resultfolder)
    mkdir(resultfolder);
end
if ~isfile(resultdatapath)
    save(resultdatapath, 'exp_date', "-v7.3");
end
% check what is already in the results file
disp(resultfile)
who('-file', resultdatapath);
end