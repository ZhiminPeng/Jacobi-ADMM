clc;clear;
load data_A.dat
load data_b.dat
load data_xs.dat

m = length(data_b);
n = length(data_xs);
k = 0.05 * n;

A = reshape(data_A, 1000, 2000);

b = data_b;
xs = data_xs;


maxit = 10000;
tol = -1; %max(5e-8,0.1*sigma);
N = 1;

idx = cell(N,1);
for i=1:N 
    % indices of i-th block
    idx{i}= (i-1)*n/N+1:i*n/N;
end

%% Proximal Jacobi ADMM
opts0.rho = 10/norm(b,1);
opts0.gamma = 1;
opts0.maxit = maxit;
opts0.tol = tol;
opts0.tau = (0.1*N*opts0.rho)*ones(N,1);
opts0.record = false;
opts0.xTrue = xs;
[x,~,Out0] = BP_ProxJADMM(A,b,idx,opts0);


semilogy(1:length(Out0.relerr), Out0.relerr, 1:maxit, res(:,1));

