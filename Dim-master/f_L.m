function f=f_L(x)
% f = sum(x,2) + 0.25*(sum(x,2)).^2 + 0.025*(sum(x,2)).^3;
 f = exp(0.3*sum(x,2));
end