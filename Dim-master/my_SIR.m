function [OrthoDir] = my_SIR(NumIn,NumPoints,SampleIn,SampleQoi)
% %test the algorithm
% A = normrnd(0,1,2,100000);
% B = (A(1,:)+A(2,:))*0.5;
% SampleQoi = exp(B);
% NumIn = 2;
% NumPoints = 100000;
% SampleIn = A;

% slice inverse regression
% global NumIn NumPoints SampleIn SampleQoi OrthoDir

% Input:
%   NumIn: original dimension
%   NumPoints: sample size
%   SampleIn: sample
%   SampleQoi: quantity of interest, according to sample

% Output:
%   OrthoDir: rotation matrix after SIR


%% SIR begins
H = 3;% slice number
num = NumPoints/H;% slice size
m = mean(SampleIn,2);% sample mean

Sigma = zeros(NumIn,NumIn);
for i = 1:NumPoints
    Sigma = Sigma+(SampleIn(:,i)-m)*(SampleIn(:,i)-m)';
end
Sigma = Sigma/NumPoints;

[~,order] = sort(SampleQoi);%slicing

Gamma = zeros(NumIn,NumIn);
for i = 1:H
    mh = zeros(NumIn,1);
    for j = 1:num
        mh = mh+SampleIn(:,order((i-1)*num+j));
    end
    mh = mh/num;
    Gamma = Gamma+(mh-m)*(mh-m)';
end
Gamma = Gamma/NumPoints*num;
[V,~] = eig(Gamma,Sigma);
OrthoDir = V(:,NumIn:-1:1);

% O = OrthoDir
% Vector = O(:,1);
% eta = transpose(Vector)*SampleIn;
% x = [-2:0.01:2];
% plot(eta,SampleQoi,'x',x,exp(x),'.');