- RWMH.m and RWMH2.m produce the same results, the only difference is verbose, since RWMH2.m diplay partial results during process.

- Optim_loop.m simulate data and estimates the model 'ns' times. First it generates data with Gamma distribution, construct the candidate density for the MH algorithm and try to optimize the 'post' function (CES) with BFGS algorithm. In the second step, tt 'try' this optimization because depending on the simulated data, this process may fail. If the optimization works, we check if the Hessian is positive definite and compute the error between the original gamma parameters and the mode parameters. After this second step, the MH algorithm process begins. Before the running s replications of the the RWMH algorithm, we test the c value (considering \Omega = c*\Simga), chosing the smallest c value in the vector c0 that returns and acceptance rate between 0.17 and 0.25. We run the Metropolis-Hastings algorithm s times, using posterior mode and Hessian at mode, and finally we compute the difference between estimated parameters and the orignal gammas, and sotre the elapsed time of RWMH function.

- Main2.m is the main script. It uses RWMH.m and optim_loop.m.

- Main1_example.m is an example script with verbose functions. It only demonstrates one iteration of optim_loop.m function with RWMH2.m. It is illustrative and return figures for the recursive averages and posterior marginal densities.

Author: Filipe Stona
Date: 20/09/2017