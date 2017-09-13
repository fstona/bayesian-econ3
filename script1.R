

rm(list = ls())
source('post.r')
source('ppred.r')
source('norm_rnd.r')
source('bw.r')
source('Mode.r')

# Sim new data
n = 200;
k = 3;
x = matrix(1,n,k);
gam = matrix(c(1.01, 0.6, 0.9, 1.2),nrow=4);
x[,2] = rchisq(n,10);
x[,3] = rchisq(n,5);
epsl = rnorm(n,0,1);
y = matrix(0,n,1);

for(i in 1:n){
  y[i,1] = gam[1] + (gam[2]*x[i,2]^gam[4] + gam[3]*x[i,3]^gam[4])^(1/gam[4]) + epsl[i];
}

#use already simulated data
data = read.csv('simdata.csv',header = F,sep = ",")
y = matrix(data[,1],ncol = 1);
x = data[,2:4];
x = as.matrix(x);
k = ncol(x);
n = nrow(x);
# gam = [1.01; 0.6; 0.9; 1.2];

ts.plot(y)

# OLS estimates
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
c=1;
vscale= c*postvar;

#set up things so first candidate draw is always accepted
lpostdraw = -9e+200;
bdraw=bmode;

#store all draws in the following matrices
#initialize them here
b_= NULL;
skew_= NULL;
kurt_= NULL;
skewstar_= NULL;
kurtstar_= NULL;

#Specify the number of replications
s=25000;
pswitch=0;

for(i in 1:s){
  bcan = matrix(bdraw,ncol=1) + norm_rnd(vscale)
  
  lpostcan = as.numeric(bw(bcan,y,x,n));
  #log of acceptance probability
  laccprob = lpostcan-lpostdraw;
  
  #if(!is.na(laccprob)){
    if(log(runif(1,min = 0,max = 1)) < laccprob){
      lpostdraw=lpostcan;
      bdraw=bcan;
      pswitch=pswitch+1;
    }
  #}
  
 
  b_ = cbind(b_, bdraw);
  
  #now do posterior predictive p-values
  #calculate f(X,gamma)
  fgamma=matrix(0,n,1);
  for(ii in 1:2){
    fgamma = fgamma + bdraw[ii+1]*(x[,ii+1]^bdraw[4]);
  }
  fgamma = bdraw[1]*matrix(1,n,1)+fgamma^(1/bdraw[4]);
  s12 = t(y-fgamma)%*%(y-fgamma)/n;
  
  #calculate skew and kurt stats for observed data
  skewkurt = ppred(bdraw,y,fgamma,n);
  skew_ = cbind(skew_, skewkurt[1,1]);
  kurt_= cbind(kurt_, skewkurt[1,2]);
  #Simulate an artificial data set
  ystar = fgamma + sqrt(s12)*rt(n,n);
  skewkurt = ppred(bdraw,ystar,fgamma,n);
  skewstar_ = cbind(skewstar_, skewkurt[1,1]);
  kurtstar_= cbind(kurtstar_, skewkurt[1,2]);
}

alldraws = matrix(t(b_),ncol = 4);
hist(alldraws[,4]);
mean(alldraws[,4]);
Mode(alldraws[,4])

mskew=mean(t(skew_));
mkurt=mean(t(kurt_))

