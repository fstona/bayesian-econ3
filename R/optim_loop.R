optim_loop = function(gam,n,k,ns,s = 25000,c0){
  
  sim_err = NULL;
  sim_et = NULL;
  mh_err = NULL;
  mh_et = NULL;
  srun = 0;
  
  for(i in 1:ns){
    tic = Sys.time();
    x = matrix(1,n,k);
    
    for(hh in seq(from = 2, to = k, by = 2)){
      x[,hh] = rgamma(n, shape = 10, rate = 1); 
    }
    for(hh in seq(from = 3, to = k, by = 2)){
      x[,hh] = rgamma(n, shape = 5, rate = 1);
    }
    epsl = rnorm(n, mean = 0, sd = 1);
    
    yp = matrix(0,n,1);
    for(j in 1:k-1){
      yp = yp + gam[j+1]*(x[,j+1]^gam[1]);
    }
    yp = yp^(1/gam[4]);
    y = yp + gam[1]*matrix(1,n,1) + epsl;
    
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
    
    res = try(optim(parm,post,x=x,y=y,n=n,hessian=TRUE))
    
    if(class(res) != "try-error"){
      postvar = solve(res$hessian);
      a = try(chol(postvar))
      if(class(a) == "try-error"){
        a = matrix(1,4,4);
      }
      
      if(all(a != 1) & rcond(a) > .Machine$double.eps){
        bmode = res$par;
        err = bmode - gam;
        sim_err = cbind(sim_err, err);
        et = as.numeric(Sys.time() - tic);
        srun = srun+1;
        
        for (jj in c0){
          ss = 10000;
          print(jj)
          part_MH = RWMH(y,x,n,bmode, postvar, ss, jj);
          acceptancerate = part_MH$acceptancerate;
          if(acceptancerate > .17 & acceptancerate < 0.25){
            c = jj;
            break
          }
        } #end for c0
        MH_res = RWMH(y,x,n,bmode, postvar, s, c)
        err2 = colMeans(MH_res$Gsim) - gam;
        mh_err = cbind(mh_err, err2);
        mh_et = cbind(mh_et, MH_res$elapsedtime[[1]]);
        
        
      } # if(all(a != 1) & rcond(a) < .Machine$double.eps)
    } #optim error
    
  } # for ns
  mh_err = rbind(mh_err, mh_et);
  sim_err = rbind(sim_err, sim_et);
  
  return(list("mh_err" = mh_err, "sim_err" = sim_err, "srun" = srun))
  
}
