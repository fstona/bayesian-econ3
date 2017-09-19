function [prior] = priorgam(parm)
%  The priors considered are:
%  1) gam_1 is NORMAL with mean 1 and sd .5
%  2) gam_2 is UNIFORM with [.2,.99)
%  3) gam_3 is UNIFORM with [.2,.99)
%  4) gam_4 is BETA with alpha 10 and beta 2
%

paraA = 1;
paraB = .5;
P1 = normpdf(parm(1),paraA,paraB);

a = [0.2, 0.2];
b = [.99, .99];

P2 = 1/(b(1)-a(1));
P3 = 1/(b(2)-a(2));
%P3 = 1/(b(3)-a(3));

paraA = 6;
paraB = 3;

P4 = gampdf(parm(4),paraA,paraB);

f = P1*P2*P3*P4;
prior = log(f);

end

