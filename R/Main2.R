
# Housekeeping and add functions
rm(list = ls())
source('optim_loop.R')
source('post.r')
source('bw.r')
source('priorgam.R')
source('RWMH.R')
require(mvtnorm)

ns = 10

## Data: simulation parameters
n = 200;
k = 3;

gam = matrix(c(0.90, 0.40, 0.60, .90),nrow=4)

## Test MH algorithm ns times
s = 25000;
c0 = seq(from=0.2, to = 3, by = 0.2)

res_optim = optim_loop(gam,n,k,ns,s, c0)
colMeans(t(res_optim$mh_err))
colMeans(t(res_optim$sim_err))

