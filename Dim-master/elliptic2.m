
function res = elliptic2(x, xi)
    
%    ay = @(y) (1 + 0.3*xi)*ones(size(y));
    
    ay = @(y) y*(sum(xi));
    
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