workspace()

cd("D:\\PNY\\Doutorado\\Julia\\Econ3")
include("b_fun.jl")

using Distributions
using Plots
using b_fun
using Optim
using DataFrames

# Simulate new data
# n = 200;
# k = 3;
# x = ones(n,k);
# fgamma=zeros(n,1);
gam = [0.90; 0.40; 0.60; .85];
# x[:,2] = rand(Gamma(10,1),n); #rand(Chisq(10),n);
# x[:,3] = rand(Gamma(5,1),n); #rand(Chisq(5),n);
# epsl = rand(Normal(0,1),n);
# y = zeros(n,1);
#
# for i = 1:n
# 	y[i,1] = gam[1] + (gam[2]*x[i,2]^gam[4] + gam[3]*x[i,3]^gam[4])^(1/gam[4]) + epsl[i];
# end

# Use already simulated data
data_frame = readtable("simdata.csv", separator = ',', header = false);
y = data_frame[:,1];
x = data_frame[:,2:4];
n = size(x,1);
k = size(x,2);

x = convert(Array, x);
y = convert(Array, y);

# OLS
bols = inv(x'x)x'y;
s2 = (y-x*bols)'*(y-x*bols)/(n-k);
sse=(n-k)*s2;
bolscov = s2.*inv(x'*x);
bolssd=zeros(k,1);
for i = 1:k
  bolssd[i,1]=sqrt(bolscov[i,i]);
end

# Calculate posterior mode and Hessian at mode
nparam=k+1;
parm = ones(nparam,1);
parm[1:k,1]=bols;
parm = vec(parm);

opt = Optim.Options(f_tol = 1e-8, iterations = 1000);
Optim.after_while!{T}(d, state::Optim.BFGSState{T}, method::BFGS, options) = global invH = state.invH

res = optimize(p -> post(p,y,x,n), parm, BFGS(), opt);

bmode = Optim.minimizer(res);
postvar = Hermitian(invH);

#candidate generating density is Normal with mean = oldraw
#and variance matrix vscale
#experiment with different values for c and dof
c=1;
vscale= c*postvar;

#set up things so first candidate draw is always accepted
lpostdraw = -9e+200;
bdraw = bmode;

#store all draws in the following matrices
#initialize them here
b_= Array{Float64,2}(4,0);
skew_=Array{Float64,2}(1,0);
kurt_=Array{Float64,2}(1,0);
skewstar_=Array{Float64,2}(1,0);
kurtstar_=Array{Float64,2}(1,0);

#Specify the number of replications
s=25000;
pswitch=0;

#Start the loop
@time for i = 1:s

    bcan = bdraw + rand(MvNormal(vscale), 4)[:,1]

    lpostcan = bw(bcan,y,x,n);

    #log of acceptance probability
    laccprob = lpostcan-lpostdraw;

    #accept candidate draw with log prob = laccprob, else keep old draw
    if log(rand(Uniform()))<laccprob
        lpostdraw=lpostcan;
        bdraw=bcan;
        pswitch=pswitch+1;
    end
    b_ = [b_ bdraw];

    #now do posterior predictive p-values
    #calculate f(X,gamma)
    fgamma=zeros(n,1);
    for ii = 1:2
        fgamma = fgamma + bdraw[ii+1,1]*(x[:,ii+1].^bdraw[4,1]);
    end
    fgamma = bdraw[1,1]*ones(n,1)+fgamma.^Complex(1/parm[4]);
    s12 = (y-fgamma)'*(y-fgamma)/n;

    #calculate skew and kurt stats for observed data
    skewkurt = ppred(bdraw,y,fgamma,n);
    skew_ = [skew_ skewkurt[1,1]];
    kurt_= [kurt_ skewkurt[1,2]];
    #Simulate an artificial data set
    ystar = fgamma + sqrt.(s12).*rand(TDist(n),n);
    skewkurt = ppred(bdraw,ystar,fgamma,n);
    skewstar_ = [skewstar_ skewkurt[1,1]];
    kurtstar_= [kurtstar_ skewkurt[1,2]];
end

alldraws = b_';
mean(alldraws[:,1])
mode(alldraws[:,1])
histogram(alldraws[:,1])
