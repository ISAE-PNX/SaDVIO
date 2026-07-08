addpath(genpath("."))

gitpath = fullfile("D:\m.von-arnim\Documents\gitlab\");
addpath(genpath(fullfile(gitpath, "b-priot\import_functions\import_ros2bag")))
addpath(genpath(fullfile(gitpath, "b-priot/import_functions/import_Novatel/")))
addpath(genpath(fullfile(gitpath, "b-priot/import_functions/import_uBlox/")))
addpath(fullfile(gitpath, "m-vonarnim\multipath-analysis\matlab_src\import_helpers\"))
addpath(genpath(fullfile(gitpath, "m-vonarnim\gnss-estimators\functions\helpers\")))
addpath(genpath(fullfile(gitpath, "m-vonarnim\gnss-estimators\functions\gnss\convenience\")))
clear gitpath