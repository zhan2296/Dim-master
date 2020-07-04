% compute (w^2-c^2)tan(wT)-2cw
function y=fw(w,c,T)
    y=tan(w.*T)-2*c.*w./(w.^2-c^2);