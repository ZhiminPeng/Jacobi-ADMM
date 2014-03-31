% Solve the exchange probelm:
%
%   Minimize    f_1(x_1)+ ... + f_N(x_N)
%   subject to  x_1 + ... + X_N = 0
%
% where f_i(x_i)=0.5*||C_i*x_i-d_i||^2.
%------------------------------------------------------------------
% Jacobi ADMM with proximal terms 0.5*tau(i)*||x_i-x_i^k||^2
% added to x_i-subproblems
% Reference: W. Deng, M.-J. Lai, Z. Peng, and W. Yin, Parallel Multi-Block 
% ADMM with o(1/k) Convergence, UCLA CAM 13-64, 2013.
%------------------------------------------------------------------
function [X,lambda,Out] = ExchgQuad_ProxJADMM(C,d,opts)
% Get parameters
rho = opts.rho;      % penalty parameter
gamma = opts.gamma;  % relaxation factor for dual update
tau = opts.tau;      % proximal paramter
maxit = opts.maxit;  % max number of iterations
tol = opts.tol;      % tolerance
n = size(C{1},2);    % length of x_i
N = length(C);       % number of x_i's

% Start clock
t0 = tic;

% Initialization
X = zeros(n,N);
Xp = [];             % previous X
lambda = zeros(n,1);
sumX = zeros(n,1);   % sum of x_i's
sumXp = [];          % previous sumX
invIpCC = cell(N,1); % inverse of (tau(i)+1)*rho*I+Ci'*Ci
Ctd = cell(N,1);     % C_i'*d_i
for i = 1:N
    % compute inverse of (tau(i)+1)*rho*I+Ci'*Ci
    invIpCC{i} = inv((tau(i)+rho)*eye(n)+C{i}'*C{i});
    % compute C_i'*d_i
    Ctd{i} = C{i}'*d{i};
end

% Objective values
Out.objValue = zeros(maxit,1);
% Residuals
Out.residual = zeros(maxit,1);
% Count the number of tau update
count = 0; 

% Main iteration
for iter = 1:maxit
    % Store previous two iterations
    Xpp = Xp; Xp = X;
    sumXpp = sumXp; sumXp = sumX;
    
    % Solve each x_j-subproblem
    for j = 1:N
        X(:,j) = invIpCC{j}*(Ctd{j}+lambda+(tau(j)+rho)*X(:,j)-rho*sumX);
    end
    
    % Sum of x_j (residual)
    sumX = sum(X,2);
    % Update dual variable
    lambda = lambda - rho*gamma*sumX;
    
    Out.objValue(iter) = ObjVal(X);
    Out.residual(iter) = norm(sumX);
    
    % Stopping criterion
    if norm(X-Xp,'fro') < tol*norm(X,'fro')
        break
    end
    
    % Compute 2/gamma*(lambda^k-lambda^{k+1})'*A(x^k-x^{k+1})
    cross_term = 2*rho*sumX'*(sumXp-sumX);
    % Compute ||x^k-x^{k+1}||^2_G
    dx_nrm = sum((X-Xp).^2)*(tau+rho);
    % Compute (2-gamma)/(rho*gamma^2)*||lambda^k-lambda^{k+1}||^2
    dlam_nrm = (2-gamma)*rho*norm(sumX)^2;
    % Compute the lower bound of error decrease: h(u^k,u^{k+1})
    lower_bnd = dx_nrm + dlam_nrm + cross_term;
    
    % Adaptive update of proximal parameter
    if lower_bnd < 0 
        % Increase tau and redo iteration
        tau = tau*2;
        count = count + 1;
        fprintf('-- Iter %4d: tau updated --\n',iter);
        lambda = lambda + rho*gamma*sumX;
        X = Xp; Xp = Xpp;
        sumX = sumXp; sumXp = sumXpp;
        % Update invIpCC
        for i = 1:N
            invIpCC{i} = inv((tau(i)+rho)*eye(n)+C{i}'*C{i});
            Ctd{i} = C{i}'*d{i};
        end
%     elseif mod(iter,10) == 0
%         tau = tau*0.5;
%         % update invIpCC
%         for i = 1:N
%             invIpCC{i} = inv((tau(i)+rho)*eye(n)+C{i}'*C{i});
%             Ctd{i} = C{i}'*d{i};
%         end
    end    
    
    Out.objValue(iter) = ObjVal(X);
    Out.residual(iter) = norm(sumX);
end
Out.CPUtime = toc(t0);
Out.iter = iter;
Out.objValue = Out.objValue(1:iter);
Out.residual = Out.residual(1:iter);
Out.tauUpdate = count;

% Nested function: compute objective value
    function obj = ObjVal(X)
        obj = 0;
        for jj = 1:N
            obj = obj+0.5*norm(C{jj}*X(:,jj)-d{jj})^2;
        end
    end  
end


    