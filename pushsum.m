function Opt_PS = pushsum(n, N, iter_num, lambda, aalpha, features, labels, A,Adj,D, big_L,bs,K,gc, degree)
Opt_PS = zeros(iter_num-1,1);
Constraint_PS = zeros(iter_num-1,1);
w = zeros(N*n,iter_num);
x = zeros(N*n,iter_num);
y = zeros(N,iter_num);
z = zeros(N*n,iter_num);
x(:,1) = randn(N*n,1);
y(:,1) = ones(N,1);
gradient_matrix = zeros(n,N);

for iter  = 2 : iter_num
    step_size = 1;
    for ii = 1 : N
        w((ii-1)*n+1:ii*n,iter) = 0;
        y(ii,iter) = 0;
        for jj = 1 : N
            if (Adj(ii,jj) == 1)||(ii==jj)
                w((ii-1)*n+1:ii*n,iter) = w((ii-1)*n+1:ii*n,iter) + x((jj-1)*n+1:jj*n,iter-1) / (D(jj*n,jj*n)+1);
                y(ii,iter) = y(ii,iter) + y(jj,iter-1) / (D(jj*n,jj*n)+1);
            end
        end
        z((ii-1)*n+1:ii*n,iter) = w((ii-1)*n+1:ii*n,iter)/y(ii,iter);
        
        gradient = zeros(N*n,1);
        for jj=(ii-1)*bs+1:ii*bs
            gradient((ii-1)*n+1:ii*n) = gradient((ii-1)*n+1:ii*n) + gc(z((ii-1)*n+1:ii*n,iter-1),lambda,aalpha, features(:,jj), labels(jj),bs, N);
        end
        gradient_matrix(:,ii) = gradient((ii-1)*n+1:ii*n);
        x((ii-1)*n+1:ii*n,iter) = w((ii-1)*n+1:ii*n,iter) - step_size * gradient((ii-1)*n+1:ii*n);
    end
    
    Constraint_PS(iter-1,1) =  norm(A*z(:,iter))^2;
    full_grad = sum(gradient_matrix,2);
    Opt_PS(iter-1,1) = norm(full_grad)^2 + Constraint_PS(iter-1)*big_L/N^2;
end
