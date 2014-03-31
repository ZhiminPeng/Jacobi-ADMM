% Solve the exchange probelm:
%
%   Minimize    f_1(x_1)+ ... + f_N(x_N)
%   subject to  x_1 + ... + X_N = 0
%
% where f_i(x_i)=0.5*||C_i*x_i-d_i||^2.
%-----------------------------------------------------
clear;clc

%seed = 2014; % use fixed seed
seed = sum(100*clock); % use clock seed
fprintf('Seed = %d\n',seed);
RandStream.setGlobalStream(RandStream('mt19937ar','seed',seed));

% Problem size
n = 100;     % length of x_i
m = 80;      % length of d_i
N = 100;     % number of x_i's
tol = -1;    % tolerance
maxit = 100; % max number of iterations
nrun = 10;    % number of runs

% Record residuals
Res_ProxJADMM = zeros(maxit,nrun);
Res_VSADMM = zeros(maxit,nrun);
Res_CorrJADMM = zeros(maxit,nrun);
% Record objective values
Obj_ProxJADMM = zeros(maxit,nrun);
Obj_VSADMM = zeros(maxit,nrun);
Obj_CorrJADMM = zeros(maxit,nrun);

% Run test
for j = 1:nrun 
    fprintf('----- test %4i -----\n', j);
    %% Generate data (C,x,d)
    X0 = randn(n,N);
    X0(:,N) = -sum(X0(:,1:N-1),2);
    C = cell(N,1);
    d = cell(N,1);
    for i = 1:N
        C{i} = randn(m,n);
        d{i} = C{i}*X0(:,i);
    end
    
    %% Proximal Jacobi ADMM
    opts1.rho = 0.01;
    opts1.gamma = 1;
    opts1.tau = 0.1*(N-1)*opts1.rho*ones(N,1);
    opts1.maxit = maxit;
    opts1.tol = tol;
    [X,~,Out1] = ExchgQuad_ProxJADMM(C,d,opts1);
    err = norm(X-X0,'fro')/norm(X0,'fro');
    fprintf('Prox-JADMM: iter = %4i, relative error = %e\n',...
        Out1.iter,err)
    Obj_ProxJADMM(:,j) = Out1.objValue;
    Res_ProxJADMM(:,j) = Out1.residual;
    
    %% Variable Splitting ADMM
    opts2.rho = 1;
    opts2.maxit = maxit;
    opts2.tol = tol;
    [X,~,Out2] = ExchgQuad_VSADMM(C,d,opts2);
    err = norm(X-X0,'fro')/norm(X0,'fro');
    fprintf('VSADMM    : iter = %4i, relative error = %e\n',...
        Out2.iter,err)
    Obj_VSADMM(:,j) = Out2.objValue;
    Res_VSADMM(:,j) = Out2.residual;
    
    %% Jacobi ADMM with correction step
    opts3.rho = 0.01;
    opts3.gamma = 1;
    opts3.maxit = maxit;
    opts3.tol = tol;
    [X,~,Out3] = ExchgQuad_CorrJADMM(C,d,opts3);
    err = norm(X-X0,'fro')/norm(X0,'fro');
    fprintf('Corr-JADMM: iter = %4i, relative error = %e\n',...
        Out3.iter,err)
    Obj_CorrJADMM(:,j) = Out3.objValue;
    Res_CorrJADMM(:,j) = Out3.residual;

    %% Ming-jun Lai's new G-S ADMM
%     opts4.beta = 0.1;
%     opts4.maxit = maxit;
%     opts4.tol = tol;
%     [X,~,Out4] = ExchgQuad_LaiADMM(C,d,opts4);
%     err = norm(X-X0,'fro')/norm(X0,'fro');
%     fprintf('Lai-ADMM: iter = %4i, relative error = %e\n',...
%         Out4.iter,err)
    
    %% Plot results
    figure(1);
    % Plot objective values
    subplot(1,2,1), semilogy(1:Out1.iter, Out1.objValue,'b-');hold on
    semilogy(1:Out2.iter, Out2.objValue,'k-');
    semilogy(1:Out3.iter, Out3.objValue,'m-.'); 
    %semilogy(1:Out4.iter, Out4.objValue,'g-');
    hold off
    xlabel('Iteration');
    ylabel('Objective Value');
    legend('Prox-JADMM','VSADMM','Corr-JADMM')
    % Plot residuals
    subplot(1,2,2), semilogy(1:Out1.iter, Out1.residual,'b-');hold on    
    semilogy(1:Out2.iter, Out2.residual,'k-');
    semilogy(1:Out3.iter, Out3.residual,'m-.');
    %semilogy(1:Out4.iter, Out4.residual,'g-');
    hold off
    xlabel('Iteration');
    ylabel('Residual');
    legend('Prox-JADMM','VSADMM','Corr-JADMM')
end
%return
%% Plot average results
figure(2);
lw = 2; % set line width
% Plot objective values
subplot(1,2,1);
semilogy(1:maxit, geomean(Obj_ProxJADMM,2),'b-','LineWidth',lw);hold on
semilogy(1:maxit, geomean(Obj_VSADMM,2),'k-','LineWidth',lw);
semilogy(1:maxit, geomean(Obj_CorrJADMM,2),'m-.','LineWidth',lw);hold off
xlabel('Iteration','FontSize',12);
ylabel('Objective Value','FontSize',12);
legend('Prox-JADMM','VSADMM','Corr-JADMM')

% Plot residuals
subplot(1,2,2);
semilogy(1:maxit, geomean(Res_ProxJADMM,2),'b-','LineWidth',lw);hold on
semilogy(1:maxit, geomean(Res_VSADMM,2),'k-','LineWidth',lw);
semilogy(1:maxit, geomean(Res_CorrJADMM,2),'m-.','LineWidth',lw); hold off
xlabel('Iteration','FontSize',12);
ylabel('Residual','FontSize',12);
legend('Prox-JADMM','VSADMM','Corr-JADMM')

% Save data
clear X0 C d X Out X2 Out2 err;
%save ExchgQuad.mat