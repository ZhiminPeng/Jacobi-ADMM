% Solve the exchange probelm:
%
%   Minimize    f_1(x_1)+ ... + f_N(x_N)
%   subject to  x_1 + ... + X_N = 0
%
% where f_i(x_i)=0.5*||C_i*x_i-d_i||^2.
%------------------------------------------------------------------
% Jacobi ADMM with correction steps
% Reference: B. S. He, L. S. Hou, and X. M. Yuan, On full Jacobian decomposition of
% the augmented lagrangian method for separable convex programming, (2013).
%------------------------------------------------------------------
function [X,lambda,Out] = ExchgQuad_CorrJADMM(C,d,opts)
% Get parameters
rho = opts.rho;
gamma = opts.gamma;
maxit = opts.maxit;
tol = opts.tol;
n = size(C{1},2);  % length of x_i
N = length(C);     % number of x_i's

t0 = tic;

% Initialization
X = zeros(n,N);
sumX = zeros(n,1);
Xt = zeros(n,N);     % predictor
lambda = zeros(n,1);
invIpCC = cell(N,1); % inverse of (rho*I+Ci'*Ci)
Ctd = cell(N,1);     % C_i'*d_i
for i = 1:N
    invIpCC{i} = inv(rho*eye(n)+C{i}'*C{i});
    Ctd{i} = C{i}'*d{i};
end
Out.objValue = zeros(maxit,1);
Out.residual = zeros(maxit,1);

% Main iteration
for iter = 1:maxit
    % Store solution at previous iteration
    Xp = X;
    
    % Predictor: Xt, lt (output of Jacobi ADMM)
    for j = 1:N
        % Solve X-subproblems
        Xt(:,j) = invIpCC{j}*(Ctd{j}+lambda+rho*X(:,j)-rho*sumX);
    end
    sumXt = sum(Xt,2);
    % Update lambda
    lt = lambda - rho*sumXt;
    
    % Compute correction stepsize
    G_nrm = rho*norm(sumX-sumXt)^2+rho*norm(X-Xt,'fro')^2+norm(lambda-lt)^2/rho;
    alpha = (lambda-lt)'*(sumX-sumXt)*2/G_nrm+1;
    alpha = alpha*gamma;
    
    % Correction step
    X = (1-alpha)*X+alpha*Xt;
    lambda = (1-alpha)*lambda+alpha*lt;
    sumX = sum(X,2);
    
    Out.objValue(iter) = ObjVal(X);
    Out.residual(iter) = norm(sumX);
    
     % Stopping criterion
    if norm(X-Xp,'fro') < tol*norm(X,'fro')
        break
    end
end
Out.CPUtime = toc(t0);
Out.iter = iter;
Out.objValue = Out.objValue(1:iter);
Out.residual = Out.residual(1:iter);

% Nested function: compute objective value
    function obj = ObjVal(X)
        obj = 0;
        for jj = 1:N
            obj = obj+0.5*norm(C{jj}*X(:,jj)-d{jj})^2;
        end
    end  
end


    