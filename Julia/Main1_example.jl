#workspace()
#cd("D:\\PNY\\Doutorado\\Julia\\Econ3")

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

Array(Float64,1)
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

opt = Optim.Options(f_tol = 1e-8, iterations = 1000, extended_trace = true, store_trace = true);
#Optim.after_while!{T}(d, state::Optim.BFGSState{T}, method::BFGS, options) = global invH = state.invH

res = optimize(p -> post(p,y,x,n), parm, BFGS(), opt);

postvar = Hermitian(res.trace[end].metadata["~inv(H)"]);
bmode = Optim.minimizer(res);
#postvar = Hermitian(invH);

## Random Walk Metropolis-Hastings Algorithm
Nsim = 10000
c0 = 1

Gsim, logposterior, acceptancerate, elapsedtime = RWMH(y,x,n,bmode, postvar, Nsim, c0);

acceptancerate
mean(Gsim,1)
histogram(Gsim[:,4])
