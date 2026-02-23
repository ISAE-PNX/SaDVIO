%% configurable stuff
datadir = fullfile("D:/m.von-arnim/Documents/data/");
%%
[file, location] = uigetfile("*.csv", "Select a Results file", datadir);
if location
    datadir = location;
end
%%
results = readtable(fullfile(location, file), ...
    FileType="text", ...
    ReadVariableNames=true);

resultsm = readmatrix(fullfile(location, file), ...
    FileType="text");
%%
timestamps_ns = resultsm(:,1);
t_wf = resultsm(:,end-11:end);

T_wf = permute(reshape(t_wf', 4, 3, size(t_wf,1)), [2 1 3]);

T_wf_se = se3(T_wf(:,1:3,:), squeeze(T_wf(:,4,:))');
%%
figure
n_samples = 400;
plotTransforms(T_wf_se(round(linspace(1,length(T_wf_se),n_samples))))
title(file, Interpreter="none")