clc;
clear;
a = 0;
b = 1;

Nx = 20;
[x, w] = lgwt(Nx, a, b);
x = sort(x);

Dimension = 3;
NumPoints = 300;
Sample_raw = randn(Dimension, NumPoints);
Qoi = zeros(length(x), NumPoints);

for j = 1:NumPoints
        Qoi(:,j) = kdv(x, Sample_raw(:,j));
end

plot(x, mean(Qoi,2), 'o');