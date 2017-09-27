priorgam = function(parm){
  #  The priors considered are:
  #  1) gam_1 is NORMAL with mean 1 and sd .5
  #  2) gam_2 is UNIFORM with [.2,.99)
  #  3) gam_3 is UNIFORM with [.2,.99)
  #  4) gam_4 is BETA with alpha 10 and beta 2
  #
  
  paraA = 1
  paraB = .5
  P1 = pnorm(parm[1],mean = paraA, sd = .5)
  
  a = matrix(c(0.20, 0.20),1)
  b = matrix(c(0.99, 0.99),1)
  
  P2 = 1/(b[1]-a[1])
  P3 = 1/(b[2]-a[2])
  
  paraA = 6
  paraB = 3
  
  P4 = pgamma(q = parm[4], shape = paraA, rate = paraB)

  f = P1*P2*P3*P4;
  prior = log(f);
  return(prior);
}