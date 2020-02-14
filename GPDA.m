function Opt_GPDA = GPDA(x_temp, edge_index,iter_num,big_L,WW,min_eig_L_hat, A,B,D,Adj,degree,n,N,gc,lambda,aalpha, features, labels,bs)
Opt_GPDA = zeros(iter_num-1,1);
x = x_temp;
mu = zeros((edge_index-1)*n,iter_num);

for iter  = 2 : iter_num
    beta = 80 * big_L * max(max(eig(WW)), 1) / (min(min_eig_L_hat, 1)*N);
    s_temp = zeros(size(A, 1)/n, 1);
    k=1;
    for ii = 1:size(Adj,1)
        for jj = ii+1:size(Adj,1)
            if Adj(ii, jj)==1
                s_temp(k) =  1/sqrt(degree(ii)*degree(jj));
                k=k+1;
            end
        end
    end
    sigma   = diag(beta * s_temp);
    sigma   = kron(sigma, eye(n));
    
    % calculate the gradient
    gradient = zeros(N*n,1);
    gradient_matrix = zeros(n,N);
    for ii = 1 : N
        for jj=(ii-1)*bs+1:ii*bs
            gradient((ii-1)*n+1:ii*n) = gradient((ii-1)*n+1:ii*n) + gc(x((ii-1)*n+1:ii*n,iter-1),lambda,aalpha, features(:,jj), labels(jj),bs, N);
        end
        gradient_matrix(:,ii) = gradient((ii-1)*n+1:ii*n);
    end
    
    % update x and mu
    x(:,iter) =  inv( A'*sigma *A + B'*sigma *B+ beta * eye(size(D))) * (( B'* sigma * B + beta*eye(size(D))) * x(:,iter-1) - gradient - A.'*mu(:,iter-1));
    mu(:,iter) = mu(:,iter-1) + sigma*  A *x(:,iter);
    
    % calculate opt
    full_grad = sum(gradient_matrix,2);
    Opt_GPDA(iter-1,1) = norm(full_grad)^2  + norm(A*x(:,iter))^2*big_L/N^2;
end
