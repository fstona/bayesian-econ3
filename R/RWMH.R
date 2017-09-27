RWMH = function(y,x,n,bmode, postvar, s=25000, c=1){
  # Random Walk Metropolis-Hastings Algorithm
  ## candidate generating density is Normal with mean = oldraw
  ## and variance matrix vscale
  
  burn = (0.5*s)+2;
  burn = round(burn);
  
  vscale= c*postvar;
  
  ## Initialize by taking a draw from a distribution centered at mode
  k = length(bmode);
  Gsim   = matrix(0,s+1,k);
  
  go_on = FALSE;
  enough = 0;
  while(go_on == FALSE){
    Gc = t(rmvnorm(1, mean = bmode, sigma = vscale, method = "chol"));
    go_on = (Gc[4]<=1);
    enough = enough + 1;
    if(enough == 500){
      Gc[4] = 1;
      go_on = TRUE;
    }
  }

  
  Gsim[1,] = Gc;
  
  accept        = 0;
  acceptancerate = NULL;
  obj           = bw(Gsim[1,],y,x,n) + priorgam(Gsim[1,]);
  logposterior  = rep(obj,s+1);
  
  tic = Sys.time()
  # Start the loop
  for(i in 1:s){
    Gc = t(rmvnorm(1, mean = Gsim[1,], sigma = vscale, method = "chol"));
    CheckBounds = (Gc[4]<=1);
    
    if(CheckBounds == TRUE){
      prioc = priorgam(Gc);
      likic = bw(Gc,y,x,n);
      objc  = prioc+likic;
      
      if(objc == -Inf){
        Gsim[i+1,] = Gsim[i,];
        logposterior[i+1] = obj;
      } else {
        alpha = min(1,exp(objc-obj));
        u = runif(1);
        
        if(u[1] <= alpha){
          Gsim[i+1,]        = Gc;
          accept            = accept+1;
          obj               = objc;
          logposterior[i+1] = objc;
        } else{
          Gsim[i+1,] = Gsim[i,];
          logposterior[i+1] = obj;
        }
      }
    } else{
      Gsim[i+1,] = Gsim[i,];
      logposterior[i+1] = obj;
    }# end checkbound
    acceptancerate = accept/i;
  } # end for
  
  Gsim         = Gsim[burn:nrow(Gsim),];
  logposterior = logposterior[burn:length(logposterior)];
  
  elapsedtime = as.numeric(Sys.time() - tic);
  
  return(list("Gsim" = Gsim, "logposterior" = logposterior, "acceptancerate" = acceptancerate, "elapsedtime" = elapsedtime))
}