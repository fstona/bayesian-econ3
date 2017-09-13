norm_rnd = function(sig){
  h = chol(sig);
  size_row = nrow(sig);
  x = matrix(rnorm(n = size_row),ncol=1);
  y = t(h)%*%x;
  return(y)
}