module optim_module

export optim_loop

using Distributions
using Optim
using b_fun


function optim_loop(gam,n,k,ns,s = 25000,c0 = 1)
    opt = Optim.Options(f_tol = 1e-8, iterations = 1000, extended_trace = true, store_trace = true);

    sim_err = [];
    sim_et = [];
    mh_err = [];
    mh_et = [];
    srun = 0;

    for i = 1:ns
        tic();
        x = ones(n,k);

        for hh = 2:2:k
            x[:,hh] = rand(Gamma(10,1),n);
        end
        for hh = 3:2:k
            x[:,hh] = rand(Gamma(5,1),n);
        end
        epsl = rand(Normal(0,1),n);

        yp = zeros(n,1);
        for j = 1:k-1
            yp = yp + gam[j+1]*(x[:,j+1].^gam[1]);
        end
        yp = yp.^(1/gam[4]);
        y = yp + gam[1]*ones(n,1) + epsl;

        # OLS
        bols = inv(x'x)x'y;
        s2 = (y-x*bols)'*(y-x*bols)/(n-k);
        sse=(n-k)*s2;
        bolscov = s2.*inv(x'*x);
        bolssd=zeros(k,1);
        for i = 1:k
          bolssd[i,1]=sqrt(bolscov[i,i]);
        end

        # Calculate posterior mode and Hessian at mode
        nparam=k+1;
        parm = ones(nparam,1);
        parm[1:k,1]=bols;
        parm = vec(parm);

        opt = Optim.Options(f_tol = 1e-8, iterations = 1000, extended_trace = true, store_trace = true);
        res = optimize(p -> post(p,y,x,n), parm, BFGS(), opt);

        postvar = Hermitian(res.trace[end].metadata["~inv(H)"]);
            try
                a = chol(postvar);
                global a
            catch
                a = ones(4,4);
                global a
            end
        if all(a .!= 1) .& (1/cond(a)>eps())
            bmode = Optim.minimizer(res);
            err = bmode - gam;
            push!(sim_err, err);
            push!(sim_et, toq());
            srun += 1;
            c = 0;
            for jj = c0
                ss = 10000;
                _, _, acceptancerate, _ = RWMH(y,x,n,bmode, postvar, ss, jj);
                if acceptancerate>.17 && acceptancerate<.25
                    c = jj
                    break
                end
            end # for c0
            Gsim, _, _, et = RWMH(y,x,n,bmode, postvar, s, c);
            err2 = mean(Gsim,1) - gam';
            push!(mh_err, err2);
            push!(mh_et, et);
        else
            println("Erro no if all...")
        end # if all(a .== 1) & (1/cond(a)>eps())
        #y = 0;
        #gc();

    end #for ns

    mh_err = [mh_err; mh_et];
    sim_err = [sim_err; sim_et];
    return mh_err, sim_err, srun
end #fun optim_loop


end #module
