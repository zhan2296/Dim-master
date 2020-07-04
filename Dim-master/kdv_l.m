function res = kdv_l(x_value, xi_value)

    global L c xi lambda w

    x = x_value;
    xi = xi_value;
    L = 1.0;
    c = 0.3;
    N = 200;

    [w, lambda] = KL_eigenvalue(1.0/c,L,N,1e-8);

    ratio = 0.2;
    d = 0;
    cutoff = 0;

    for i = 2:N
        if lambda(i) <= ratio*lambda(1) && cutoff == 0
            d = i;
            cutoff = 1;
        end
    end
    
    lambda = lambda(1:5);
    A = zeros(d,1);
    B = zeros(5,1);
    
    format long
    for i = 1:d
       phi = @(y) KL_fun_generator(y, lambda(i), w(i), c, L);
       A(i) = integral(phi,0,1)*xi(i);
    end
    B(1) = integral(@(t) temp_fun(t),0,1)*xi(1);
    B(2) = integral(@(t) temp_fun2(t),0,1)*xi(2);
    B(3) = integral(@(t) temp_fun3(t),0,1)*xi(3);
    B(4) = integral(@(t) temp_fun4(t),0,1)*xi(4);
    B(5) = integral(@(t) temp_fun5(t),0,1)*xi(5);
    
    ep = 1;
    res = 1 + ep*sum(A) - 2*sech(x - 4 + ep*6*sum(B)).^2;
    
end