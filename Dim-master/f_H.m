function f=f_H(x)
f = 0.7*f_L(x) + 0.5*exp(0.5*(x(:,1) + x(:,3)));
% f = 0.7*f_L(x) + 0.8*sum(x,2);
% f = 0.9*f_L(x) + 0.3;
end