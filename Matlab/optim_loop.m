function [mh_err, sim_err, srun] = optim_loop(gam,n,k,ns,opts,s,c0)

mh_err = [];
sim_err = [];
sim_et = [];
mh_et = [];
srun = 0;

for i = 1:ns
    tic()
    x = ones(n,k);
    
    for hh = 2:2:k
        x(:,hh) = gamrnd(10,1,n,1);%wblrnd(5,10,n,1);%chi2rnd(10,n,1);%lognrnd(1,.5,n,1);%
    end
    for hh = 3:2:k
        x(:,3) = gamrnd(5,1,n,1);%wblrnd(5,5,n,1);%chi2rnd(5,n,1);%lognrnd(1,.5,n,1);%
    end
    epsl = normrnd(0,1,n,1);
    
    yp=zeros(n,1);
    for j = 1:k-1
        yp = yp + gam(j+1,1)*(x(:,j+1).^gam(end,1));
    end
    yp = yp.^(1/gam(end,1));
    y = yp + gam(1,1)*ones(n,1) + epsl;
    
    %% Calculate posterior mode and Hessian at mode
    % Constructing the Candidate Density for MH Algorithm
    
    %OLS estimates
    bols = inv(x'*x)*x'*y;
    s2 = (y-x*bols)'*(y-x*bols)/(n-k);
    sse=(n-k)*s2;
    bolscov = s2*inv(x'*x);
    bolssd=zeros(k,1);
    for i = 1:k
        bolssd(i,1)=sqrt(bolscov(i,i));
    end
    
    nparam=k+1;
    parm = ones(nparam,1);
    parm(1:k,1)=bols;
    
    try [bmode,~,~,~,~,hess] = fminunc(@(p)post(p,y,x,n),parm,opts);
        
        try a = chol(inv(hess));
        catch a = ones(4);
        end
        
        if (all(all(a)) ~= 1) && (rcond(inv(hess))>eps)
            err = bmode - gam;
            sim_err = [sim_err err];
            et = toc();
            sim_et = [sim_et et];
            srun = srun+1;
            
            if s == [];
                s = 25000;
            end
            postvar = inv(hess);
            for jj = c0
                ss = 10000;
                [~, ~, acceptancerate, ~] = RWMH(y,x,n,bmode, postvar, ss, jj);
                if acceptancerate > .17 && acceptancerate < 0.25
                    c = jj;
                    break
                end
            end
            [Gsim, ~, ~, et] = RWMH(y,x,n,bmode, postvar, s, c);
            
            err2 = mean(Gsim)' - gam;
            mh_err = [mh_err err2];
            mh_et = [mh_et et];

        end
        clear y
    end
end

mh_err = [mh_err; mh_et];
sim_err = [sim_err; sim_et];

end

