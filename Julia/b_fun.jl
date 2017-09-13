module b_fun

export post
export bw
export ppred
export norm_rnd

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


end
