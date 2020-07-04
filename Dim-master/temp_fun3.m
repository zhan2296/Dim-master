function y = temp_fun3(t)
global L c xi lambda w
fun = @(y) KL_fun_generator(y, lambda(3), w(3), c, L);
y = zeros(size(t));
for k = 1:length(t)
    y(k) = integral(fun,0,t(k));
end