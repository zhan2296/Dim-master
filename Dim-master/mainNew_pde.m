function [M, A_h, A_l] = mainNew_pde()
%% Pre-processing
clc; close all;
rng('default')

addpath ./Utilities

set(0,'defaulttextinterpreter','latex')

global ModelInfo

%% Setup
N_L = 300;
N_H_start = 50;
N_H = N_H_start;
D = 5;

I = 1;
I_max = 4;
O = cell(1,I_max+1);
count = 0;
points_add = zeros(1,I_max);
process_error = zeros(1,I_max);

% lb = zeros(1,D);
% ub = ones(1,D); 
jitter = 1e-10;
noise_L = 0;
noise_H = 0;
ModelInfo.jitter=jitter;

%% Generate Data
% ModelInfo.X_H = bsxfun(@plus,lb,bsxfun(@times,   lhsdesign(N_H,D)    ,(ub-lb)));
ModelInfo.X_H = randn(N_H_start, D);
ModelInfo.y_H = zeros(N_H_start, 1);
for i = 1:N_H_start
%    ModelInfo.y_H(i) = kdv_h(6,ModelInfo.X_H(i,:));
    ModelInfo.y_H(i) = elliptic2(6,ModelInfo.X_H(i,:));
end


% ModelInfo.X_L = bsxfun(@plus,lb,bsxfun(@times,   lhsdesign(N_L,D)    ,(ub-lb)));
ModelInfo.X_L = randn(N_L, D);
ModelInfo.y_L = zeros(N_L, 1);

for i = 1:N_L
%    ModelInfo.y_L(i) = kdv_l(6, ModelInfo.X_L(i,:));
    ModelInfo.y_L(i) = elliptic1(6, ModelInfo.X_L(i,:));
end

f_H_pool = zeros(N_L, 1);
for i = 1: N_L
%    f_H_pool(i) = kdv_h(6, ModelInfo.X_L(i,:));
    f_H_pool(i) = elliptic2(6, ModelInfo.X_L(i,:));
end

n_test = 50;
X_T = randn(n_test, D);
y_exact = zeros(n_test, 1);
for i = 1:n_test
%    y_exact(i) = kdv_h(6, X_T(i,:));
    y_exact(i) = elliptic2(6, X_T(i,:));
end

n_SIR = 500;
X_SIR = randn(n_SIR, D);

X_ORIGIN = X_SIR;
low_origin = zeros(n_SIR, 1);
for i = 1:n_SIR
%    low_origin(i) = kdv_l(6, X_SIR(i,:));
    low_origin(i) = elliptic1(6, X_SIR(i,:));
end

%% First rotation
A1 = my_SIR(5, N_L, ModelInfo.X_L', ModelInfo.y_L);
O{count+1} = A1;
count = count + 1;
ModelInfo.X_H = ModelInfo.X_H * A1;
ModelInfo.X_L = ModelInfo.X_L * A1;
X_T = X_T * A1;
X_SIR = X_SIR * A1;

%% First iteration

hyp = [log(ones(1, 2*D+2)) 1 -4 -4];

% hyp = [logsigma1 logtheta1_1 logtheta1_2 logtheta1_3 logtheta1_4 ... logtheta1_rd
%        logsigma2 logtheta2_1 logtheta2_2 logtheta2_3 logtheta2_4 ... logtheta2_rd
%        rho logsigma_eps_L logsigma_eps_H]
options = optimoptions('fminunc','GradObj','on','Display','iter',...
    'Algorithm','trust-region','Diagnostics','on','DerivativeCheck','on',...
    'FinDiffType','central');
[ModelInfo.hyp,~,~,~,~,~] = fminunc(@likelihood,hyp,options);

[~, var_f] = predictor_f_H(ModelInfo.X_L);
[mean_f_test_old, ~] = predictor_f_H(X_T);
% predict_H_star = random('Normal',mean_f_H_star, var_f_H_star);
% fprintf(1,'Relative L2 error f_H: %e\n', (norm(mean_f_H_star-f_H_star,2)/norm(f_H_star,2)));

%% Active Learning for adding new f_H
relative_error = 1;
N_iteration = 1;
while  N_iteration < 5

    index = find(var_f == max(var_f),1,'first');
    X_H_new = ModelInfo.X_L(index,:);
    N_H = N_H + 1;
    ModelInfo.X_H = vertcat(ModelInfo.X_H, X_H_new);
    ModelInfo.y_H = vertcat(ModelInfo.y_H, f_H_pool(index));
    
    hyp = [log(ones(1, 2*D+2)) 1 -4 -4];
    options = optimoptions('fminunc','GradObj','on','Display','iter',...
    'Algorithm','trust-region','Diagnostics','on','DerivativeCheck','on',...
    'FinDiffType','central');
    [ModelInfo.hyp,~,~,~,~,~] = fminunc(@likelihood,hyp,options);

    [~, var_f] = predictor_f_H(ModelInfo.X_L);
    [mean_f_test_new, ~] = predictor_f_H(X_T);
    relative_error = norm(mean_f_test_new - mean_f_test_old,2)/norm(mean_f_test_old,2);
    
    if relative_error > 1e-3
        mean_f_test_old = mean_f_test_new;
        N_iteration = N_iteration + 1;
    else
        break
    end
end


%% Sample from Multi fidelity 
[mean_f_SIR, ~] = predictor_f_H(X_SIR);

%% Dimension reduction
A = my_SIR(5,n_SIR,X_SIR',mean_f_SIR);
O{count+1} = A;
count = count + 1;
A = A(:,1:5);

%% Reduced dimension multi fidelity
ModelInfo.X_H = (ModelInfo.X_H) * A;
ModelInfo.X_L = (ModelInfo.X_L) * A;
X_T = X_T * A;
X_SIR = X_SIR * A;
D_new = 5;

hyp_new = [log(ones(1, 2*D_new+2)) 1 -4 -4];
options = optimoptions('fminunc','GradObj','on','Display','iter',...
    'Algorithm','trust-region','Diagnostics','on','DerivativeCheck','on',...
    'FinDiffType','central');
[ModelInfo.hyp,~,~,~,~,~] = fminunc(@likelihood,hyp_new,options);

[mean_f_T, ~] = predictor_f_H(X_T);

% fprintf(1,'Relative L2 error f_H: %e\n', (norm(mean_f_T - y_exact,2)/norm(y_exact,2)));
% fprintf(1,'Relative L2 error f_H: %e\n', norm(mean_f_test_new - y_exact,2)/norm(y_exact,2));
% A

%% Iteration start

while I <= I_max
   
    points_add(I) = N_H - N_H_start;
    N_H_start = N_H;
    process_error(I) = norm(mean_f_T - y_exact,2)/norm(y_exact,2);
    
    if process_error(I) < 1e-3
        break
    else
        
        hyp = [log(ones(1, 2*D+2)) 1 -4 -4];

        % hyp = [logsigma1 logtheta1_1 logtheta1_2 logtheta1_3 logtheta1_4 ... logtheta1_rd
        %        logsigma2 logtheta2_1 logtheta2_2 logtheta2_3 logtheta2_4 ... logtheta2_rd
        %        rho logsigma_eps_L logsigma_eps_H]
        options = optimoptions('fminunc','GradObj','on','Display','iter',...
            'Algorithm','trust-region','Diagnostics','on','DerivativeCheck','on',...
            'FinDiffType','central');
        [ModelInfo.hyp,~,~,~,~,~] = fminunc(@likelihood,hyp,options);

        [~, var_f] = predictor_f_H(ModelInfo.X_L);
        [mean_f_test_old, ~] = predictor_f_H(X_T);
        
        relative_error = 1;
        N_iteration = 1;
        while  N_iteration < 5

            index = find(var_f == max(var_f),1,'first');
            X_H_new = ModelInfo.X_L(index,:);
            N_H = N_H + 1;
            ModelInfo.X_H = vertcat(ModelInfo.X_H, X_H_new);
            ModelInfo.y_H = vertcat(ModelInfo.y_H, f_H_pool(index));
            
            hyp = [log(ones(1, 2*D+2)) 1 -4 -4];
            options = optimoptions('fminunc','GradObj','on','Display','iter',...
            'Algorithm','trust-region','Diagnostics','on','DerivativeCheck','on',...
            'FinDiffType','central');
            [ModelInfo.hyp,~,~,~,~,~] = fminunc(@likelihood,hyp,options);

            [~, var_f] = predictor_f_H(ModelInfo.X_L);
            [mean_f_test_new, ~] = predictor_f_H(X_T);
            relative_error = norm(mean_f_test_new - mean_f_test_old,2)/norm(mean_f_test_old,2);

            if relative_error > 1e-3
                mean_f_test_old = mean_f_test_new;
                N_iteration = N_iteration + 1;
            else
                N_iteration = 20;
            end
        end
        
        [mean_f_SIR, ~] = predictor_f_H(X_SIR);
        A = my_SIR(5,n_SIR,X_SIR',mean_f_SIR);
        O{count+1} = A;
        count = count + 1;
        A = A(:,1:5);
        
        ModelInfo.X_H = (ModelInfo.X_H) * A;
        ModelInfo.X_L = (ModelInfo.X_L) * A;
        X_T = X_T * A;
        X_SIR = X_SIR * A;
        D_new = 5;

        hyp_new = [log(ones(1, 2*D_new+2)) 1 -4 -4];
        options = optimoptions('fminunc','GradObj','on','Display','iter',...
            'Algorithm','trust-region','Diagnostics','on','DerivativeCheck','on',...
            'FinDiffType','central');
        [ModelInfo.hyp,~,~,~,~,~] = fminunc(@likelihood,hyp_new,options);

        [mean_f_T, ~] = predictor_f_H(X_T);        
        I = I + 1;
    end   
end

 
M = O{1};
for i = 2:length(O)
    M = M*O{i};
end

%% New Generate
N_L = 300;
N_H = 50;
D = 5;

jitter = 1e-13;
ModelInfo.jitter=jitter;

%% Generate Data
% ModelInfo.X_H = bsxfun(@plus,lb,bsxfun(@times,   lhsdesign(N_H,D)    ,(ub-lb)));
ModelInfo.X_H = randn(N_H, D);
ModelInfo.y_H = zeros(N_H, 1);
for i =1:N_H
    ModelInfo.y_H(i) = kdv_h(6, ModelInfo.X_H(i,:));
end
XH = ModelInfo.X_H;

% ModelInfo.X_L = bsxfun(@plus,lb,bsxfun(@times,   lhsdesign(N_L,D)    ,(ub-lb)));
ModelInfo.X_L = randn(N_L, D);
ModelInfo.y_L = zeros(N_L, 1);
for i = 1:N_L
    ModelInfo.y_L(i) = kdv_l(6, ModelInfo.X_L(i,:));
end
XL = ModelInfo.X_L;

n_test = 50;
X_T = randn(n_test, D);
y_exact = zeros(n_test,1);
for i = 1:n_test
    y_exact(i) = kdv_h(6, X_T(i,:));
end
XT = X_T;

%% Reduced GP with M

ModelInfo.X_H = XH * M(:,1:2);
ModelInfo.X_L = XL * M(:,1:2);
X_T = XT * M(:,1:2);

hyp = [log(ones(1, 2*2+2)) 1 -4 -4];

options = optimoptions('fminunc','GradObj','on','Display','iter',...
    'Algorithm','trust-region','Diagnostics','on','DerivativeCheck','on',...
    'FinDiffType','central');
[ModelInfo.hyp,~,~,~,~,~] = fminunc(@likelihood,hyp,options);
[f_M, ~] = predictor_f_H(X_T);


error1 = norm(f_M - y_exact,2)/norm(y_exact,2)


%% Post-processing
                                    rmpath ./Utilities