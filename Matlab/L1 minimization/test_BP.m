clear; clc

%seed = 2014; % use fixed seed
seed = sum(100*clock); % use clock seed
fprintf('Seed = %d\n',seed);
RandStream.setGlobalStream(RandStream('mt19937ar','seed',seed));

% Problem size
m = 750;  % number of measurements
n = 2.5 *m; % solution length

k = 60;   % sparsity 
% Partition data into N blocks
N = 1;
idx = cell(N,1);
for i=1:N 
    % indices of i-th block
    idx{i}= (i-1)*n/N+1:i*n/N;
end

% standard deviation of noise
sigma = 0*1e-3;
% max number of iterations
maxit = 500;
% tolerance
tol = -1; %max(5e-8,0.1*sigma);
% number of runs
nrun = 10;

% Record solution errors 
Err_ProxJADMM = zeros(maxit,nrun);
Err_Yall1 = zeros(maxit,nrun);
Err_VSADMM = zeros(maxit,nrun);
Err_CorrJADMM = zeros(maxit,nrun);

% Run test
for j = 1:nrun
    %% Generate data
    A = randn(m,n);
    xs = zeros(n,1); % ground truth
    p = randperm(n);
    xs(p(1:k)) = randn(k,1);
    b = A*xs + sigma*randn(m,1);
    
    %% Proximal Jacobi ADMM
    opts0.rho = 10/norm(b,1);
    opts0.gamma = 1;
    opts0.maxit = maxit;
    opts0.tol = tol;
    opts0.tau = (0.1*N*opts0.rho)*ones(N,1);
    opts0.record = false;
    opts0.xTrue = xs;
    [x,~,Out0] = BP_ProxJADMM(A,b,idx,opts0);
    % relative error
    err = norm(x-xs)/norm(xs);
    fprintf('Prox-JADMM: iter = %4i, relative error = %e\n', Out0.iter,err)
    Err_ProxJADMM(:,j) = Out0.relerr;
    
    %% YALL1
    opts1.tol = tol;
    opts1.maxit = maxit;
    opts1.print = 2;
    opts1.xs = xs;
    [x,Out1] = yall1(A, b, opts1);
    err = norm(x-xs)/norm(xs);
    fprintf('YALL1: iter = %4i, relative error = %e\n', Out1.iter,err)
    Err_Yall1(:,j) = Out1.relerr;

    %% Variable Splitting ADMM
    opts2.rho = 10/norm(b,1);
    opts2.maxit = maxit;
    opts2.tol = tol;
    for i = 1:N
        opts2.tau(i) = 1.01*opts2.rho*norm(A(:,idx{i}))^2;
    end
    opts2.xTrue = xs;
    [x,~,Out2] = BP_VSADMM(A,b,idx,opts2);
    err = norm(x-xs)/norm(xs);
    fprintf('VSADMM: iter = %4i, relative error = %e\n', Out2.iter,err)
    Err_VSADMM(:,j) = Out2.relerr;
    
    %% Jacobi ADMM with correction step
    opts3.rho = 10/norm(b,1);
    opts3.gamma = 1;
    opts3.maxit = maxit;
    opts3.tol = tol;
    opts3.xTrue = xs;
    [x,~,Out3] = BP_CorrJADMM(A,b,opts3);
    err = norm(x-xs)/norm(xs);
    fprintf('CorrADMM: iter = %4i, relative error = %e\n',...
        Out3.iter,err)
    Err_CorrJADMM(:,j) = Out3.relerr;

    %% Plot error curves
    figure(1);
    semilogy(1:maxit, Out0.relerr, 'b-');hold on
    semilogy(1:maxit, Out1.relerr, 'r--');
    semilogy(1:maxit, Out2.relerr, 'k-');
    semilogy(1:maxit, Out3.relerr, 'm-.');
    legend('Prox-JADMM','YALL1','VSADMM','Corr-JADMM')
    hold off
end

%return

%% Plot average result
figure(2);
lw = 2; % line width
t = 1:1:maxit;
semilogy(t, geomean(Err_ProxJADMM(t,:),2), 'b-','LineWidth',lw);hold on
semilogy(t, geomean(Err_Yall1(t,:),2), 'r--','LineWidth',lw);
semilogy(t, geomean(Err_VSADMM(t,:),2), 'k-','LineWidth',lw);
semilogy(t, geomean(Err_CorrJADMM(t,:),2), 'm-.','LineWidth',lw);
legend('Prox-JADMM','YALL1','VSADMM','Corr-JADMM')
xlabel('Iteration','FontSize',12)
ylabel('Relative Error','FontSize',12)
hold off

% Save data
clear A x xs lambda p Out0 Out1 Out2 Out3 i j b d err;
%save L1.mat