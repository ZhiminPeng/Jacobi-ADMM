% Solve the basis pursuit problem:
%   Minimize ||x||_1, subject to Ax = b,
% where A and x are divided into N blocks.
% Here, N = length of x.
%--------------------------------------------------
% Jacobi ADMM with correction steps  
% (using dynamically updated stepsizes)
% Reference: B. S. He, L. S. Hou, and X. M. Yuan, On full Jacobian decomposition of
% the augmented lagrangian method for separable convex programming, (2013).
%---------------------------------------------------
function [x,lambda,Out] = BP_CorrJADMM(A,b,opts)
% Get parameters
rho = opts.rho;
gamma = opts.gamma;
maxit = opts.maxit;
tol = opts.tol;
[m,n] = size(A);

% Initialization
x = zeros(n,1);   
xt = zeros(n,1);            % predictor
lambda = zeros(m,1);
Ax = zeros(m,n);            % [..., Ai*xi, ...]
Axt = zeros(m,n);           % [..., Ai*xt_i, ...]
res = b;                    % residual b-A*x
Out.alpha = zeros(maxit,1); % correction stepsize
t0 = tic;

if isfield(opts,'xTrue')
    x0 = opts.xTrue; % ground truth
    Out.relerr = zeros(maxit,1); % record solution errors
end 

% Squared 2-norm of each column of A
aa = zeros(n,1);
for j = 1:n
    aa(j) = sum(A(:,j).^2);
end

% Main iteration
for iter = 1:maxit
    % store solution at previous iteration
    xp = x; 
    lp = lambda;
    
    % Predictor: xt, lt (output of Jacobi ADMM)
    for j = 1:n
        c = lambda/rho + res + Ax(:,j);
        xt(j) = Shrink_L1(c'*A(:,j)/aa(j), 1/(rho*aa(j)));
        Axt(:,j) = A(:,j)*xt(j);
    end
    % Compute residual
    res_t = b-sum(Axt,2);
    % Update lambda
    lt = lambda+rho*res_t;
    
    % Compute step size
    G_norm = rho*norm(res-res_t)^2+rho*norm(Ax-Axt,'fro')^2+norm(lambda-lt)^2/rho;
    alpha = (lambda-lt)'*(res_t-res)*(2/G_norm)+1;
    alpha = alpha*gamma;
    Out.alpha(iter) = alpha;
    
    % Correction step
    x = (1-alpha)*xp + alpha*xt;
    lambda = (1-alpha)*lp + alpha*lt;
    
    % Compute A_j*x_j
    for j = 1:n
        Ax(:,j) = A(:,j)*x(j);
    end 
    % Residual
    res = b - sum(Ax,2);
    
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