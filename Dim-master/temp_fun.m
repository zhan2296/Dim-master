function y = temp_fun(t)
global L c xi lambda w
fun = @(y) KL_fun_generator(y, lambda(1), w(1), c, L);
y = zeros(size(t));
for k = 1:length(t)
    y(k) = integral(fun,0,t(k));
end