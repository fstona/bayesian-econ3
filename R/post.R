post = function(parm,y,x,n){
  # Evaluate the log of the marginal posterior for parm at a point
  
  fgamma = matrix(0,n,1);
  
  for(ii in 1:2){
    fgamma = fgamma + parm[ii+1]*(x[,ii+1]^parm[4]);
  }
  
  fgamma = fgamma^(1/parm[4]);
  fgamma = fgamma + parm[1]*matrix(1,n,1);
  
  # Ignoring non-essential constants, evaluate negative of log-posterior
  lpost = .5*n*log(t(y - fgamma)%*%(y-fgamma));
  return(lpost)
}