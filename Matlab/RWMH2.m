function [Gsim, logposterior, acceptancerate, elapsedtime] = RWMH2(y,x,n,bmode, postvar, s, c)
% Random Walk Metropolis-Hastings Algorithm
%candidate generating density is Normal with mean = oldraw
%and variance matrix vscale

if c == []
    c = 1;
end

%Specify the number of replications
if s == [];
    s=25000;
end
burn = int32(0.5*s)+2;


vscale= c*postvar;

% Initialize by taking a draw from a distribution centered at mode
k = size(bmode,1);
Gsim   = zeros(s,k);

go_on = 0;
enough = 0;
while go_on == 0
   Gc = mvnrnd(bmode',c*vscale);
   go_on = (Gc(4)<=1);  % bounds
   enough = enough+1;
   if enough == 500
       Gc(4) = 1;
       go_on = 1;
       warning('Go on loop break')
   end
end
Gsim(1,:) = Gc;

accept        = 0;
obj           = bw(Gsim(1,:)',y,x,n) + priorgam(Gsim(1,:));
counter       = 0;
logposterior  = obj*ones(s,1);

%Start the loop
tic
for i = 1:s

    Gc = mvnrnd(Gsim(i,:),c*vscale);
    %bcan=bdraw + norm_rnd(vscale);
    CheckBounds = (Gc(4)<=1);
    
    if CheckBounds == 1
        prioc = priorgam(Gc);
        likic = bw(Gc',y,x,n);
        objc  = prioc+likic;
        if objc == -Inf
            
            Gsim(i+1,:) = Gsim(i,:);
            logposterior(i+1) = obj;
            
        else % objc > -Inf
            
            alpha = min(1,exp(objc-obj));
            u = rand(1);
            
            if u<=alpha
                Gsim(i+1,:)   = Gc;
                accept            = accept+1;
                obj               = objc;
                logposterior(i+1) = objc;
            else
                Gsim(i+1,:)   = Gsim(i,:);
                logposterior(i+1) = obj;
            end
            
        end % if objc == -Inf
        
    else % CheckBounds NE 1
        
       Gsim(i+1,:) = Gsim(i,:);
       logposterior(i+1) = obj;
       
    end  % if CheckBounds == 1

    acceptancerate     = accept/i;
    counter            = counter + 1;

    if counter==5000
       disp('                                                                  ');
       disp(['                               DRAW NUMBER:', num2str(i)]         );
       disp('                                                                  ');
       disp('                                                                  ');    
       disp(['                           ACCEPTANCE RATE:', num2str(acceptancerate)]);
       disp('                            RECURSIVE AVERAGES                    ');
       disp('                                                                  ');
       disp('Gamma_1     Gamma_2     Gamma_3     Gamma_4');
       disp(num2str(mean(Gsim(1:i,:))));  
       disp('                                                                  ');

       counter = 0;
    end % if counter==500

end %for i=1:s
    
Gsim    = Gsim(burn:end,:);
logposterior= logposterior(burn:end);

disp('                                                                  ');
disp(['                     ELAPSED TIME:   ', num2str(toc)]             );

elapsedtime=toc;



end

