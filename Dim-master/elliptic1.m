
function res = elliptic1(x_value, xi_value)
    
    global L c xi lambda w

    x = x_value;
    xi = xi_value;
    L = 1.0;
    c = 0.3;
    N = 200;

    [w, lambda] = KL_eigenvalue(1.0/c,L,N,1e-8);

    ratio = 0.1;
    d = 0;
    cutoff = 0;

    for i = 2:N
        if lambda(i) <= ratio*lambda(1) && cutoff == 0
            d = i;
            cutoff = 1;
        end
    end
   
    ay = @(y) 0.1 + exp(0.2*(KL_fun_generator(y, lambda(1), w(1), c, L)*xi(1) + KL_fun_generator(y, lambda(2), w(2), c, L)*xi(2)));
    
    format long
    ay_inv1 = @(z) 1./ay(z);
    ay_inv2 = @(z) z./ay(z);
   
    temp1 = integral(ay_inv1,0,1);
    temp2 = integral(ay_inv2,0,1);
    temp = temp2/temp1;
    
    u = @(z) (temp - z)./ay(z);
    
    res = zeros(1,length(x));
    
    for i = 1:length(x)
        res(i) = integral(u, 0, x(i),'AbsTol', 1e-12);
    end

end