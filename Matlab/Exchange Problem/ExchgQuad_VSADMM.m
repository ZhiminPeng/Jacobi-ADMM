% Solve the exchange probelm:
%
%   Minimize    f_1(x_1)+ ... + f_N(x_N)
%   subject to  x_1 + ... + X_N = 0
%
% where f_i(x_i)=0.5*||C_i*x_i-d_i||^2.
%------------------------------------------------------------------
% Exchange ADMM (see Boyd et al., Distributed Optimization and Statistical
% Learning via the Alternating Direction Method of Multipliers)
%------------------------------------------------------------------
function [X,u,Out] = ExchgQuad_VSADMM(C,d,opts)
% Get parameters
rho = opts.rho;
maxit = opts.maxit;
tol = opts.tol;
n = size(C{1},2); % length of x_i
N = length(C); % number of x_i's

t0 = tic;

% Initialization
X = zeros(n,N);
u = zeros(n,1);      % dual variable
X_sum = zeros(n,1);  % sum of x_i's
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
    
    % Solve each x_j-subproblem
    for j = 1:N
        X(:,j) = invIpCC{j}*(Ctd{j}+rho*(X(:,j)-X_sum/N-u));
    end
    X_sum = sum(X,2);
    % Update dual variable
    u = u + X_sum/N;
    
    Out.objValue(iter) = ObjVal(X);
    Out.residual(iter) = norm(X_sum);
    
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


    