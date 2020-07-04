function y = temp_fun2(t)
global L c xi lambda w
fun = @(y) KL_fun_generator(y, lambda(2), w(2), c, L);
y = zeros(size(t));
for k = 1:length(t)
    y(k) = integral(fun,0,t(k));
end