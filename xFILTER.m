function [Opt, Q] = xFILTER(D, x, edge_index,iter_num,big_L, A, n,N,gc,lambda,aalpha, features, labels,bs)

% Parameter Chosen
% % I
k=1;
Lnorm = (D)^(-1/2) * (A' * A) * (D)^(-1/2);
beta = 96 * k * D * big_L / sum(sum(D));
eig_Lnorm = eig(Lnorm);
for ii = 1 : length(eig_Lnorm)
    if (eig_Lnorm(ii)>=1e-10)
        min_eig_Lnorm = eig_Lnorm(ii);
        break;
    end
end
sigma = 48* 96 * big_L/ (k*sum(sum(D))*min_eig_Lnorm);

% % % % II
% beta = 96*big_L/N;
% Lnorm = A' * A  ;
% eig_Lnorm = eig(Lnorm);
% for ii = 1 : length(eig_Lnorm)
%     if (eig_Lnorm(ii)>=1e-10)
%         min_eig_Lnorm = eig_Lnorm(ii);
%         break;
%     end
% end
% sigma = 48* beta / min_eig_Lnorm;

% Iterative Update
mu = zeros((edge_index-1)*n,iter_num);
Opt = zeros(iter_num-1,1);
for iter  = 2 : iter_num
    
    % Calculate the gradient
    gradient = zeros(N*n,1);
    for ii = 1 : N
        for jj=(ii-1)*bs+1:ii*bs
            gradient((ii-1)*n+1:ii*n) = gradient((ii-1)*n+1:ii*n) + gc(x((ii-1)*n+1:ii*n,iter-1),lambda,aalpha, features(:,jj), labels(jj),bs, N);
        end
    end
    gradient_matrix = reshape(gradient, n, N);
    
    % Update x, and mu
    R = beta^(-1) * (sigma * (A' * A)) + eye(N*n);
    d = x(:,iter-1) - beta^(-1) * gradient - beta^(-1) * A' * mu(:,iter-1);
    [x(:,iter), Q] = Chebyshev(R, d, x(:,iter-1), N, beta);
    mu(:,iter) = mu(:,iter-1) + sigma * (A*x(:,iter));
    
    % Calculate opt
    full_grad = sum(gradient_matrix,2);
    Opt(iter-1,1) =  norm(full_grad)^2  + norm(A*x(:,iter))^2*big_L/N^2;
    
end
