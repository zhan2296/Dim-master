function mainNew()
%% Pre-processing
clc; close all;
rng('default')

addpath ./Utilities

set(0,'defaulttextinterpreter','latex')

global ModelInfo

%% Setup
N_L = 100;
N_H_start = 5;
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
jitter = 1e-9;
noise_L = 0;
noise_H = 0;
ModelInfo.jitter=jitter;

%% Generate Data
% ModelInfo.X_H = bsxfun(@plus,lb,bsxfun(@times,   lhsdesign(N_H,D)    ,(ub-lb)));
ModelInfo.X_H = randn(N_H_start, D);
ModelInfo.y_H = f_H(ModelInfo.X_H);
ModelInfo.y_H = ModelInfo.y_H + noise_H*randn(N_H,1);


% ModelInfo.X_L = bsxfun(@plus,lb,bsxfun(@times,   lhsdesign(N_L,D)    ,(ub-lb)));
ModelInfo.X_L = randn(N_L, D);
ModelInfo.y_L = f_L(ModelInfo.X_L);
ModelInfo.y_L = ModelInfo.y_L + noise_L*randn(N_L,1);

n_test = 50;
X_T = randn(n_test, D);
y_exact = f_H(X_T);

n_SIR = 1000;
X_SIR = randn(n_SIR, D);

%% First rotation
A1 = my_SIR(5, 50, ModelInfo.X_L', ModelInfo.y_L);
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
while  N_iteration < 10

    index = find(var_f == max(var_f),1,'first');
    X_H_new = ModelInfo.X_L(index,:);
    N_H = N_H + 1;
    ModelInfo.X_H = vertcat(ModelInfo.X_H, X_H_new);
    ModelInfo.y_H = f_H(ModelInfo.X_H);
    ModelInfo.y_H = ModelInfo.y_H + noise_H*randn(N_H,1);
    
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
A = my_SIR(5,1000,X_SIR',mean_f_SIR);
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
        while  N_iteration < 10

            index = find(var_f == max(var_f),1,'first');
            X_H_new = ModelInfo.X_L(index,:);
            N_H = N_H + 1;
            ModelInfo.X_H = vertcat(ModelInfo.X_H, X_H_new);
            ModelInfo.y_H = f_H(ModelInfo.X_H);
            ModelInfo.y_H = ModelInfo.y_H + noise_H*randn(N_H,1);

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
        A = my_SIR(5,1000,X_SIR',mean_f_SIR);
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


%% Post-processing
rmpath ./Utilities