function fun = KL_fun_generator(x, lambda, w, c, L)
   fun = sqrt(lambda)*(w/c*cos(w*x)+sin(w*x))./sqrt(1/2*(1+w^2/c^2)*L + (w^2/c^2 - 1)*sin(2*w*L)/4/w + 1/2/c*(1 - cos(2*w*L)));
%      fun = (x.^i + 1)*xi;
end