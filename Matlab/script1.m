

%% Housekepping
clear all;
clc;

% Add path of data and functions
addpath(genpath('G:\Matlab Codes\Econ3 - bayesian\'));

%% Data
%sim
n = 200;
k = 3;
x = ones(n,k);
gam = [1.01; 0.6; 0.9; 1.2];
x(:,2) = chi2rnd(10,n,1);
x(:,3) = chi2rnd(5,n,1);
epsl = normrnd(0,1,n,1);

for i = 1:n
	y(i,1) = gam(1) + (gam(2)*x(i,2)^gam(4) + gam(3)*x(i,3)^gam(4))^(1/gam(4)) + epsl(i);
end

load simdata.dat;
y = simdata(:,1);
x = simdata(:,2:end);
[n k] = size(x);

%% Calculate posterior mode and Hessian at mode

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

[bmode,fval,exitflag,output,grad,hess] = fminunc(@(p)post(p,y,x,n),parm,opts);

postvar = inv(hess);
chol(postvar)
%% RWMH

%candidate generating density is Normal with mean = oldraw
%and variance matrix vscale
%experiment with different values for c and dof
c=1;
vscale= c*postvar;


%set up things so first candidate draw is always accepted
lpostdraw = -9e+200;
bdraw=bmode;

%store all draws in the following matrices
%initialize them here
b_=[];
skew_=[];
kurt_=[];
skewstar_=[];
kurtstar_=[];

%Specify the number of replications
s=25000;
pswitch=0;
%Start the loop
tic
for i = 1:s
    
    bcan=bdraw + norm_rnd(vscale);
    
    lpostcan = bw(bcan,y,x,n);
    
    %log of acceptance probability
    laccprob = lpostcan-lpostdraw;
    
    %accept candidate draw with log prob = laccprob, else keep old draw
    if log(rand)<laccprob
        lpostdraw=lpostcan;
        bdraw=bcan;
        pswitch=pswitch+1;
    end
    b_ = [b_ bdraw];
    
    %now do posterior predictive p-values
    %calculate f(X,gamma)
    fgamma=zeros(n,1);
    for ii = 1:2
        fgamma = fgamma + bdraw(ii+1,1)*(x(:,ii+1).^bdraw(4,1));
    end
    fgamma = bdraw(1,1)*ones(n,1)+fgamma.^(1/bdraw(4,1));
    s12 = (y-fgamma)'*(y-fgamma)/n;
    
    %calculate skew and kurt stats for observed data
    skewkurt = ppred(bdraw,y,fgamma,n);
    skew_ = [skew_ skewkurt(1,1)];
    kurt_= [kurt_ skewkurt(1,2)];
    %Simulate an artificial data set
    ystar = fgamma + sqrt(s12)*trnd(n,n,1);
    skewkurt = ch5ppred(bdraw,ystar,fgamma,n);
    skewstar_ = [skewstar_ skewkurt(1,1)];
    kurtstar_= [kurtstar_ skewkurt(1,2)];
    
end
toc
alldraws = [b_'];
hist(alldraws(:,4)) % gam4 = 1.2
mode(alldraws(:,4))

mskew=mean(skew_');
mkurt=mean(kurt_');

skstar = abs(skewstar_');
skstar=sort(skstar);
for ii = 1:s
if abs(mskew)<skstar(ii,1)
    break
end
end
skewpval=1-ii/s;

kstar = abs(kurtstar_');
kstar=sort(kstar);
for ii = 1:s
if abs(mkurt)<kstar(ii,1)
    break
end
end
kurtpval=1-ii/s;
kurtpval


hist(skewstar_',25)
title('Figure 5.1: Posterior Predictive Density for Skewness')
xlabel('Skewness')

%hist(kurtstar_',25)
%title('Figure 5.2: Posterior Predictive Density for Kurtosis')
%xlabel('Kurtosis')
