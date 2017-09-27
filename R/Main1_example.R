## Estimation of CES function with Bayesian Inferance
# This script is based on Koop (2003) and Herbst and Schorfheide (2015),
# taking advantage of computer codes from Gary Koop and Minsu Chang, which
# can be found in the respective websites: 
#      - Bayesian Econometrics (Koop, 2003): http://www.wiley.com/legacy/wileychi/koopbayesian/
#      - Bayesian Estimation of DSGE Models (Herbst and Schorfheide, 2015): https://web.sas.upenn.edu/schorf/companion-web-site-bayesian-estimation-of-dsge-models/
#
# After cleaning, we simulate data (or load already simulated data), than
# we estimate a linear model with OLS and optmize the CES function with a
# BFGS algorithm. We use the posterior at mode and the Hessian at mode as
# canditade density.
# 
#
# Author: Filipe Stona
# Date: 19/09/2017

# Housekeeping and add functions
rm(list = ls())
source('post.r')
source('ppred.r')
source('norm_rnd.r')
source('bw.r')
source('Mode.r')
source('priorgam.R')
source('RWMH.R')
require(mvtnorm)

## Sim new data
n = 200;
k = 3;
x = matrix(1,n,k);
gam = matrix(c(0.90, 0.40, 0.60, .85),nrow=4);
x[,2] = rchisq(n,10);
x[,3] = rchisq(n,5);
epsl = rnorm(n,0,1);
y = matrix(0,n,1);

for(i in 1:n){
  y[i,1] = gam[1] + (gam[2]*x[i,2]^gam[4] + gam[3]*x[i,3]^gam[4])^(1/gam[4]) + epsl[i];
}

## use already simulated data
data = read.csv('simdata.csv',header = F,sep = ",")
y = matrix(data[,1],ncol = 1);
x = data[,2:4];
x = as.matrix(x);
k = ncol(x);
n = nrow(x);
gam = matrix(c(0.90, 0.40, 0.60, .85),nrow=4);

## OLS estimates
bols = solve(t(x)%*%x)%*%t(x)%*%y;
s2 = t(y-x%*%bols)%*%(y-x%*%bols)/(n-k);
sse=(n-k)%*%s2;
bolscov = matrix(s2,3,3)*solve(t(x)%*%x);
bolssd=matrix(0,k,1);
for(i in 1:k){
  bolssd[i,1]=sqrt(bolscov[i,i]);
}

nparam=k+1;
parm = matrix(1,nparam,1);
parm[1:k,1]=bols;

parm  = as.vector(parm)

res = optim(parm,post,x=x,y=y,n=n,hessian=TRUE)


## RWMH
bmode = res$par;
postvar = solve(res$hessian);
chol(postvar)

#candidate generating density is Normal with mean = oldraw
#and variance matrix vscale
#experiment with different values for c and dof

Nsim = 50000;
c0 = 1;
MH_res = RWMH(y,x,n,bmode, postvar, Nsim, c0);

MH_res$elapsedtime[[1]]
colMeans(MH_res$Gsim)
hist(MH_res$Gsim[,3])
