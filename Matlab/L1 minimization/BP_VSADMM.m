% Solve the basis pursuit problem:
%   Minimize ||x||_1, subject to Ax = b,
% where A and x are divided into N blocks.
%----------------------------------------------------------
% Variable Splitting ADMM: solve
%   Min  ||x||_1,
%   s.t. A_i*x_i-b/N = y_i, sum_i y_i = 0.
%
% x-subproblem: solved approximately by prox-linear method
%(see Boyd et al., Distributed Optimization and Statistical
% Learning via the Alternating Direction Method of Multipliers)
%----------------------------------------------------------
function [x,lambda,Out] = BP_VSADMM(A,b,idx,opts)
% Get parameters
rho = opts.rho;
maxit = opts.maxit;
tau = opts.tau;
tol = opts.tol;
[m,n] = size(A);
N = length(idx); % # of blocks

% Initialization
x = zeros(n,1);
lambda = cell(N,1);
y = cell(N,1);
Ax = cell(N,1);
grad = cell(N,1);
for i=1:N 
    lambda{i} = zeros(m,1); 
    y{i} = zeros(m,1);
    Ax{i} = zeros(m,1);
end
b = b/N;

t0 = tic;

if isfield(opts,'xTrue')
    x0 = opts.xTrue; % ground truth
    Out.relerr = zeros(maxit,1); % store solution errors
end

% Main iteration
for iter = 1:maxit
    % Store previous x
    xp = x;
    
    % Compute residual: sum_i A_i*x_i-b/N-lambda_i/rho
    res = zeros(m,1);
    for i = 1:N
        res = res + Ax{i} - b - lambda{i}/rho;
    end
    
    % Block update
    for i = 1:N
        % update y
        y{i} = Ax{i} - b - lambda{i}/rho-res/N;
        % compute gradient
        grad{i} = A(:,idx{i})'*res*(rho/N);
        % compute x_i approximately by prox-linear method
        x(idx{i}) = Shrink_L1(x(idx{i})-grad{i}/tau(i),1/tau(i));
        % compute A_i*x_i
        Ax{i} = A(:,idx{i})*x(idx{i});
        % update dual variable
        lambda{i} = lambda{i}-rho*(Ax{i}-b-y{i});
    end
    
    % Stopping criterion
    if norm(x-xp) < tol*norm(x)
        break;
    end
       
    % Compute relative error
    if isfield(opts,'xTrue')
        Out.relerr(iter) = norm(x-x0)/norm(x0);
    end

end
Out.CPUtime = toc(t0);
Out.iter = iter;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% L1-shrinkage for Min r||z||_1+1/2*||z-x||^2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = Shrink_L1(x,r)
x = sign(x).*max(abs(x)-r,0);
end