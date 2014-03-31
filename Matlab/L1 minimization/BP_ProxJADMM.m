% Solve the basis pursuit problem:
%   Minimize ||x||_1, subject to Ax = b,
% where A and x are divided into N blocks.
%---------------------------------------------------
% Jacobi ADMM with prox-linear terms
% Reference: W. Deng, M.-J. Lai, Z. Peng, and W. Yin, Parallel Multi-Block ADMM with
% o(1/k) Convergence, UCLA CAM 13-64, 2013.
%---------------------------------------------------
function [x,lambda,Out] = BP_ProxJADMM(A,b,idx,opts)
% Get parameters
rho = opts.rho;     % penalty parameter
gamma = opts.gamma; % relaxation factor for dual update
maxit = opts.maxit; % max number of iterations
tau = opts.tau;     % proximal paramter
tol = opts.tol;     % tolerance
[m,n] = size(A);
N = length(idx);    % number of blocks

% Start clock
t0 = tic; 

% Initialization
x = zeros(n,1);
lambda = zeros(m,1);
res = -b;            % residual A*x-b
xp = [];             % previous x
res_p = [];          % previous residual


% if opts.record
%     Out.x = zeros(n,maxit);
%     Out.y = zeros(n,maxit);
%     Out.lambda = zeros(n,maxit);
% end

if isfield(opts,'xTrue')
    x0 = opts.xTrue; % ground truth
    Out.relerr = zeros(maxit,1); % record solution errors
end

% Main iteration
for iter = 1:maxit
    % Store previous two iterations
    xpp = xp; xp = x;
    res_pp = res_p; res_p = res;
    
    % Solve each x_j-subproblem by prox-linear method
    grad = A'*(rho*res-lambda);
    for j = 1:N
        % update each block
        x(idx{j}) = Shrink_L1(x(idx{j})-(1/tau(j))*grad(idx{j}),1/tau(j));
    end
    % Compute residual
    res = A*x-b;
    % Update dual variable
    lambda = lambda-rho*gamma*res;
    norm(lambda)
    % Stopping criterion
    if norm(x-xp) < tol*norm(x)
        break;
    end
    
    % Compute 2/gamma*(lambda^k-lambda^{k+1})'*A(x^k-x^{k+1})
    cross_term = 2*rho*res'*(res_p-res);
    % Compute ||x^k-x^{k+1}||^2_G
    dx_nrm = 0;
    for j = 1:N
        dx_nrm = dx_nrm + norm(x(idx{j})-xp(idx{j}))^2*tau(j);
    end
    % Compute (2-gamma)/(rho*gamma^2)*||lambda^k-lambda^{k+1}||^2
    dlam_nrm = (2-gamma)*rho*norm(res)^2;
    % Compute the lower bound of error decrease: h(u^k,u^{k+1})
    lower_bnd = dx_nrm + dlam_nrm + cross_term;
    
    % Adaptive update of proximal parameter
    if lower_bnd < 0
        % Increase tau and redo iteration
        tau = tau*2;
        fprintf('-- Iter %4d: tau updated --\n',iter);
        lambda = lambda + rho*gamma*res;
        x = xp; xp = xpp;
        res = res_p; res_p = res_pp;
    elseif mod(iter,10) == 0
        % Decrease tau after every 10 iterations
        tau = tau*0.5;
    end
    
    if opts.record
        Out.x(:,iter) = x;
        Out.y(:,iter) = y;
        Out.lambda(:,iter) = lambda;
    end
    
    % Compute relative error
    if isfield(opts,'xTrue')
        Out.relerr(iter) = norm(x-x0)/norm(x0);
        Out.obj(iter) = norm(x,1);
    end
    
end
Out.CPUtime = toc(t0); % cpu time
Out.iter = iter; % number of iterations
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% L1-shrinkage for Min r||z||_1+1/2*||z-x||^2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = Shrink_L1(x,r)
x = sign(x).*max(abs(x)-r,0);
end