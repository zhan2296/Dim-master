function y = temp_fun4(t)
global L c xi lambda w
fun = @(y) KL_fun_generator(y, lambda(4), w(4), c, L);
y = zeros(size(t));
for k = 1:length(t)
    y(k) = integral(fun,0,t(k));
end