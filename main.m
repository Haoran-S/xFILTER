clc; clear; close all;
tic
rng('default')
%% Parameters
n          = 10;   % problem dimention
batch_size = 10;   % batch size
nodes_num  = 10;   % number of agents in the network
K          = batch_size * nodes_num; % number of data points
repeat_num = 1;    % number of trials
iter_num   = 5000; % number of iterations per trial
radius     = 0.5;

gc = @(x,lambda,alpha,z,y,bs, M) 1/(bs*M)*(-y * z)/(1+exp(y*x.'*z))+1/(M)*((2*lambda*alpha*x)./((1+alpha*x.^2).^2)); % gradient

function_lambda = 0.001;
function_aalpha = 1;

%% Initialization
Opt_GPDA = zeros(iter_num-1,repeat_num);
Opt_Chebyshev = zeros(iter_num-1,repeat_num);
Opt_DSG = zeros(iter_num-1,repeat_num);
Opt_PS = zeros(iter_num-1,repeat_num);
Opt_NEXT = zeros(iter_num-1,repeat_num);

%% Data
features = randn(n,K);
labels= randi([1,2], 1, K); labels(labels==2) = -1; % labels \in {-1,1}
features_norm = features/norm(features,'fro');
big_L=1/(batch_size)*norm(features_norm,'fro')^2+2*function_lambda*function_aalpha*n;

%% Algorithms
for repeat_index = 1 : repeat_num
    disp(repeat_index);
    [Adj, degree, num_of_edge,A,B,D,Lm,edge_index, eig_Lm,min_eig_Lm,WW,LN,L_hat,eig_L_hat,min_eig_L_hat] = Generate_Graph(nodes_num,radius,n);
    
    y_temp = zeros(nodes_num*n,iter_num);
    y_temp(:,1) = randn(nodes_num*n,1);
    
    [Opt_Chebyshev(:,repeat_index), Q2] = xFILTER(D, y_temp, edge_index,iter_num,big_L,  A, n,nodes_num,gc,function_lambda,function_aalpha, features, labels,batch_size);
    Opt_GPDA(:,repeat_index) = GPDA(y_temp, edge_index,iter_num,big_L,WW,min_eig_L_hat, A,B,D,Adj,degree,n,nodes_num,gc,function_lambda,function_aalpha, features, labels,batch_size);
    Opt_DSG(:,repeat_index) = DSG(y_temp,  iter_num,big_L,  A, Adj, n,nodes_num,gc,function_lambda,function_aalpha, features, labels,K,batch_size, degree);
    Opt_PS(:,repeat_index) = pushsum(n, nodes_num, iter_num,  function_lambda, function_aalpha, features, labels, A,Adj,D, big_L,batch_size,K,gc, degree);
    Opt_NEXT(:,repeat_index) = NEXT(y_temp,  iter_num, big_L,  A, Adj, n,nodes_num,gc,function_lambda,function_aalpha, features, labels,K,batch_size,degree);
end

%% plot the results
linewidth = 2.5;
fontsize = 14;
MarkerSize = 10;
figure;
semilogy(mean(Opt_Chebyshev,2),'linestyle', ':', 'linewidth',linewidth,'color', 'r');hold on;
semilogy(1:Q2:size(mean(Opt_Chebyshev,2))*Q2, mean(Opt_Chebyshev,2),'linestyle', '-', 'linewidth',linewidth,'color', 'r');
semilogy(mean(Opt_GPDA,2),'linestyle', '--', 'linewidth',linewidth,'color', 'm');hold on;
semilogy(1:2:size(mean(Opt_NEXT,2))*2, mean(Opt_NEXT,2),'linestyle', '-.','linewidth',linewidth,'color', 'k');hold on;
semilogy(mean(Opt_DSG,2),'linestyle', ':', 'linewidth',linewidth,'color', 'b');hold on;
semilogy(mean(Opt_PS,2),'linestyle', '--','linewidth',linewidth,'color', 'g');hold on;

xlim([0,iter_num]);
le = legend('xFILTER (outer)', 'xFILTER (total)', 'Prox-PDA','NEXT','DSG','Push-sum');
xl = xlabel('Number of Communication Rounds','FontSize',fontsize);
yl = ylabel('Optimality Gap h^*','FontSize',fontsize);
savefig(sprintf('figure_random_nodes%d_bs%d_fea%d_iter%d',nodes_num, batch_size, n, iter_num));
toc
