%% Estimation of CES function with Bayesian Inferance
% This script is based on Koop (2003) and Herbst and Schorfheide (2015),
% taking advantage of computer codes from Gary Koop and Minsu Chang, which
% can be found in the respective websites: 
%    - Bayesian Econometrics (Koop, 2003): http://www.wiley.com/legacy/wileychi/koopbayesian/
%    - Bayesian Estimation of DSGE Models (Herbst and Schorfheide, 2015): https://web.sas.upenn.edu/schorf/companion-web-site-bayesian-estimation-of-dsge-models/
%
% After cleaning, we simulate data (or load already simulated data), than
% we estimate a linear model with OLS and optmize the CES function with a
% BFGS algorithm. We use the posterior at mode and the Hessian at mode as
% canditade density.
% 
%
% Author: Filipe Stona
% Date: 19/09/2017

%% Housekepping
clear all;
clc;
delete *.asv;

% Add path of data and functions
addpath(genpath('G:\Matlab Codes\Econ3 - bayesian\'));

%% Data
%sim
new          = input('Simulate new dataset? yes = 1 or no = 0:  ');
disp('                                                                  ');
disp('                                                                  ');
if new == 1
    n = 200;
    k = 3;
    x = ones(n,k);
    gam = [0.90; 0.40; 0.60; .85];
    x(:,2) = gamrnd(10,1,n,1);%lognrnd(1,.5,n,1);%
    x(:,3) = gamrnd(5,1,n,1);%lognrnd(1,.5,n,1);%
    epsl = normrnd(0,1,n,1);
    
    for i = 1:n
        y(i,1) = gam(1) + (gam(2)*x(i,2)^gam(4) + gam(3)*x(i,3)^gam(4))^(1/gam(4)) + epsl(i);
    end
else
    load simdata.mat;
    % simdata.mat uses:
    %   gam = [0.90; 0.40; 0.60; .85];
    %   x(:,2) = gamrnd(10,1,n,1);
    %   x(:,3) = gamrnd(5,1,n,1);

end
%% Calculate posterior mode and Hessian at mode
% Constructing the Candidate Density for MH Algorithm

%OLS estimates
bols = inv(x'*x)*x'*y;
s2 = (y-x*bols)'*(y-x*bols)/(n-k);
sse=(n-k)*s2;
bolscov = s2*inv(x'*x);
bolssd=zeros(k,1);
for i = 1:k
    bolssd(i,1)=sqrt(bolscov(i,i));
end

nparam=k+1;
parm = ones(nparam,1);
parm(1:k,1)=bols;

opts  = optimset('Display','final');
opts.MaxFunEvals = 60000;
opts.MaxIter = 1000;
opts.FunValCheck = 'on';
opts.LargeScale  = 'off';
opts.OptimalityTolerance = 1e-8;
%opts.TolFun = 1e-14;
opts.HessUpdate = 'bfgs';
opts.PlotFcns = @optimplotfval;

post(parm,y,x,n)

[bmode,fval,exitflag,output,grad,hess] = fminunc(@(p)post(p,y,x,n),parm,opts);

postvar = inv(hess);

chol(postvar)
%%  Random Walk Metropolis-Hastings Algorithm

Nsim          = input('How Many Posterior Draws?:  ');
disp('                                                                  ');
disp('                                                                  ');

c0 = 1;
[Gsim, logposterior, acceptancerate, elapsedtime] = RWMH2(y,x,n,bmode, postvar, Nsim, c0);

% Results
[mean(Gsim)' gam mean(Gsim)'-gam]
[bmode gam bmode-gam]


%=========================================================================
%                            RECURSIVE AVERAGES 
%=========================================================================


[s,np] = size(Gsim);
rmean = zeros(s,np);

for i=1:s
    rmean(i,:) = mean(Gsim(1:i,:),1);
end

pnames = strvcat('\gamma_{1}','\gamma_{2}', '\gamma_{3}','\gamma_{4}');

for i=1:np
    
subplot((np)/2,2,i), plot(rmean(:,i),'LineStyle','-','Color','b',...
        'LineWidth',2.5), hold on
title(pnames(i,:),'FontSize',12,'FontWeight','bold');    
end

%=========================================================================
%                   POSTERIOR MARGINAL DENSITIES 
%=========================================================================

figure('Position',[20,20,900,600],'Name',...
    'Posterior Marginal Densities','Color','w')


for i=1:(np)
    xmin = min(Gsim(:,i));
    xmax = max(Gsim(:,i));
    grid = linspace(xmin,xmax,100);
    u    = (1+0.4)*max(ksdensity(Gsim(:,i)));
subplot((np)/2,2,i), plot(grid,ksdensity(Gsim(:,i)),'LineStyle','-','Color','b',...
        'LineWidth',2.5), hold on
plot([mean(Gsim(:,i)) mean(Gsim(:,i))], [0 u],'LineStyle',':',...
    'Color','black','LineWidth',2.5 ), hold off
axis([xmin xmax 0 u]);
title(pnames(i,:),'FontSize',12,'FontWeight','bold');    
end

