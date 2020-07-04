% eigenfunction for kernel exp(-|t-s|/b)

function y=KL_eigenf(t,w,c,T)
    p1=w.*cos(w*t)/c+sin(w*t);
    p2=0.5*(1+w.^2/c^2)*T;
    p3=(w.^2/c^2-1).*sin(2*w*T)/4./w;
    p4=(1-cos(2*w*T))/2/c;
    y=p1./(p2+p3+p4).^0.5;
   