% clear
% clc

%% Pre-processing
clc; close all;
rng('default')

addpath ./Utilities

set(0,'defaulttextinterpreter','latex')

global ModelInfo


[M, A_h, A_l] = mainNew();

N_L = 100;
N_H = 20;
D = 5;

jitter = 1e-7;
ModelInfo.jitter=jitter;

%% Generate Data
% ModelInfo.X_H = bsxfun(@plus,lb,bsxfun(@times,   lhsdesign(N_H,D)    ,(ub-lb)));
ModelInfo.X_H = randn(N_H, D);
ModelInfo.y_H = f_H(ModelInfo.X_H);
XH = ModelInfo.X_H;

% ModelInfo.X_L = bsxfun(@plus,lb,bsxfun(@times,   lhsdesign(N_L,D)    ,(ub-lb)));
ModelInfo.X_L = randn(N_L, D);
ModelInfo.y_L = f_L(ModelInfo.X_L);
XL = ModelInfo.X_L;

n_test = 50;
X_T = randn(n_test, D);
y_exact = f_H(X_T);
XT = X_T;

%% Reduced GP with M

ModelInfo.X_H = XL * M(:,1:2);
ModelInfo.X_L = XH * M(:,1:2);
X_T = XT * M(:,2);

hyp = [log(ones(1, 2*2+2)) 1 -4 -4];

options = optimoptions('fminunc','GradObj','on','Display','iter',...
    'Algorithm','trust-region','Diagnostics','on','DerivativeCheck','on',...
    'FinDiffType','central');
[ModelInfo.hyp,~,~,~,~,~] = fminunc(@likelihood,hyp,options);
[f_M, ~] = predictor_f_H(X_T);

%% Reduced GP with AL AND AH

K = [A_h(:,1), A_l(:,1)];
ModelInfo.X_H = XL * K;
ModelInfo.X_L = XH * K;
X_T = XT * K;

hyp = [log(ones(1, 2*2+2)) 1 -4 -4];
options = optimoptions('fminunc','GradObj','on','Display','iter',...
    'Algorithm','trust-region','Diagnostics','on','DerivativeCheck','on',...
    'FinDiffType','central');
[ModelInfo.hyp,~,~,~,~,~] = fminunc(@likelihood,hyp,options);
[f_K, ~] = predictor_f_H(X_T);

error1 = norm(f_M - y_exact,2)/norm(y_exact,2)
error2 = norm(f_K - y_exact,2)/norm(y_exact,2)