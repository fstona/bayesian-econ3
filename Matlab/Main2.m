%% Comparing optimization time and accuracy

%% Housekepping
clear all;
clc;

% Add path of data and functions
addpath(genpath('G:\Matlab Codes\Econ3 - bayesian\'));

%% optim options
opts  = optimset('Display','off');
opts.MaxFunEvals = 60000;
opts.MaxIter = 1000;
opts.FunValCheck = 'on';
opts.LargeScale  = 'off';
opts.OptimalityTolerance = 1e-8;
opts.HessUpdate = 'bfgs';

ns = 10; %Numer of simulations

%% Data
%simulation parameters
n = 200;
k = 3;
gam = [0.9; 0.4; 0.6; .9];

if size(gam,1)-1 ~= k;
    error('Error: Wrong gam size')
end

% Test MH algorithm ns times
s = 25000;
c0 = .2:.2:1.2;
[all_mh, allerr, srun] = optim_loop(gam,n,k,ns,opts,s,c0);


%% Display Results

vartype     = {'\gamma_{1}','\gamma_{2}', '\gamma_{3}', '\gamma_{4}', 'elapsed time'};
      
disp('=======================================================================');
disp(' Variable Name           Mean      St. Dev.     Min        Max         ');
disp('=======================================================================');
for hh=1:length(vartype);
    fprintf('%-20s %10.4f %10.4f %10.4f %10.4f\n',vartype{hh},mean(allerr(hh,:)),...
        std(allerr(hh,:)), min(allerr(hh,:)),max(allerr(hh,:)));    
end
disp('======================================================================='); 


disp('=======================================================================');
disp(' Variable Name           Mean      St. Dev.     Min        Max         ');
disp('=======================================================================');
for hh=1:length(vartype);
    fprintf('%-20s %10.4f %10.4f %10.4f %10.4f\n',vartype{hh},mean(all_mh(hh,:)),...
        std(all_mh(hh,:)), min(all_mh(hh,:)),max(all_mh(hh,:)));    
end
disp('======================================================================='); 
