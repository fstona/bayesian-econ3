# Main2
##cd("D:\\PNY\\Doutorado\\Julia\\Econ3")
include("b_fun.jl")
include("optim_module.jl")

using Optim

ns = 10

## Data: simulation parameters
n = 200;
k = 3;

gam = [0.9; 0.4; 0.6; .9];

## Test MH algorithm ns times
s = 25000;
c0 = .2:.2:3;

mh_err, sim_err, srun = optim_module.optim_loop(gam,n,k,ns,s,c0);

hcat(sim_err[1:ns]...)
