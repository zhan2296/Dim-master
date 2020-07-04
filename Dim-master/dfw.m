%compute derivation of (w^2-c^2)tan(wT)-2cw
function y=dfw(w,c,T)
     y=T./cos(w.*T).^2+2*c*(w.^2+c^2)./(w.^2-c^2).^2;
    