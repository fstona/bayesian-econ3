module b_fun

using Distributions

export post
export bw
export ppred
export norm_rnd
export priorgam
export RWMH

function post(parm,y,x,n)
  # Evaluate the log of the marginal posterior for parm at a point

  fgamma=zeros(n,1);
  for ii = 1:2
    fgamma = fgamma + parm[ii+1]*(x[:,ii+1].^(parm[4]));
  end

  fgamma = fgamma.^Complex(1/parm[4]);
  fgamma = fgamma + parm[1]*ones(n,1);

  # Ignoring non-essential constants, evaluate negative of log-posterior
  lpost = .5*n*log.((y - fgamma)'*(y-fgamma));
  return real(lpost[1,1])
end

function bw(parm,y,x,n)
  # Evaluate the log of the marginal posterior for parm at a point

  fgamma=zeros(n,1);
  for ii = 1:2
    fgamma = fgamma + parm[ii+1]*(x[:,ii+1].^(parm[4]));
  end

  fgamma = fgamma.^Complex(1/parm[4]);
  fgamma = fgamma + parm[1]*ones(n,1);

  # Ignoring non-essential constants, evaluate negative of log-posterior
  lpost = .5*n*log.((y - fgamma)'*(y-fgamma));
  return -real(lpost[1,1])
end

function ppred(parm,ystar,fgamma,n)
  #Given a data set (either observed or simulated), evaluate skewness
  #and kurtosis stats which are used for posterior predictive p-values

  errors= ystar - fgamma;
  errors2=errors.^2;
  errors3=errors.^3;
  errors4=errors.^4;
  skew = sqrt.(n)*sum(errors3)/(sum(errors2)^1.5);
  kurt = n*sum(errors4)/(sum(errors2)^2) -3;
  skewkurt = [skew kurt];
end

function norm_rnd(sig)
  h = chol(sig);
  size_mat = size(sig,1);
  x = randn(size_mat,1);
  y = h'*x;
end

function priorgam(parm)
  #  The priors considered are:
  #  1) gam_1 is NORMAL with mean 1 and sd .5
  #  2) gam_2 is UNIFORM with [.2,.99)
  #  3) gam_3 is UNIFORM with [.2,.99)
  #  4) gam_4 is BETA with alpha 10 and beta 2
  #

  paraA = 1
  paraB = .5
  P1 = pdf(Normal(paraA,paraB),parm[1])

  a = [0.2, 0.2]
  b = [.99, .99]

  P2 = 1/(b[1]-a[1])
  P3 = 1/(b[2]-a[2])

  paraA = 6
  paraB = 3

  P4 = pdf(Gamma(paraA,paraB),parm[4])

  f = P1*P2*P3*P4
  prior = log(f)
  return prior;
end

function RWMH(y,x,n,bmode, postvar, s = 25000, c = 1)
  # Random Walk Metropolis-Hastings Algorithm
  ## candidate generating density is Normal with mean = oldraw
  ## and variance matrix vscale

  burn = (0.5*s)+2;
  burn = trunc(Int32,burn);

  vscale= c*postvar;

  ## Initialize by taking a draw from a distribution centered at mode
  k = size(bmode,1);
  Gsim   = zeros(s+1,k);

  go_on = false;
  enough = 0;
  Gc = ones(1,4);
  while go_on == false
    let Gc = Gc
      Gc[:] = rand(MvNormal(bmode,vscale), 4)[:,1]'
    end
    go_on = (Gc[4]<=1);  # bounds
    enough += 1;
    if enough == 500
      Gc[4] = 1;
      go_on = true;
    end #if enough
  end # while go_on
  Gsim[1,:] = Gc;


  accept        = 0;
  acceptancerate = [];
  obj           = bw(vec(Gsim[1,:]'),y,x,n) + priorgam(Gsim[1,:]);
  logposterior  = obj*ones(s+1,1);

  tic()
  for i = 1:s
    let Gc = Gc
      Gc[:] = rand(MvNormal(Gsim[i,:],vscale), 4)[:,1]'
    end
    CheckBounds = (Gc[4]<=1);

    if CheckBounds == true
      prioc = priorgam(Gc);
      likic = bw(Gc,y,x,n);
      objc  = prioc+likic;

      if objc == -Inf
        Gsim[i+1,:] = Gsim[i,:];
        logposterior[i+1] = obj;
      else #objc == -Inf
        alpha = min(1,exp(objc-obj));
        u = rand(1);
        if u[1] <= alpha
          Gsim[i+1,:]       = Gc[:];
          accept            += 1;
          obj               = objc;
          logposterior[i+1] = objc;
        else
          Gsim[i+1,:] = Gsim[i,:];
          logposterior[i+1] = obj;
        end #alpha <= u
      end # objc == -Inf
    else #CheckBounds != true
      Gsim[i+1,:] = Gsim[i,:];
      logposterior[i+1] = obj;
    end #if CheckBounds == true
#    let acceptancerate = acceptancerate
    acceptancerate = accept/i;
#    end
  end # for
  let Gsim = Gsim, logposterior = logposterior
    Gsim         = Gsim[burn:end,:];
    logposterior = logposterior[burn:end,:];
  end
  elapsedtime = toq();

  return Gsim, logposterior, acceptancerate, elapsedtime

end # RWMH



end #module
