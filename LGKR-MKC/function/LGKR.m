function [H_normalized, G, Z, obj] = LGKR(K,k,lambda,beta)
%initialization
sample_num = size(K,1);
kernel_num = size(K,3);
mu = ones(kernel_num,1)/kernel_num;
K_mu = combine_K_mu(K,mu);
G = zeros(sample_num,sample_num);
%G = K_mu;
Z = eye(sample_num);
%Z = zeros(sample_num,sample_num);
%Kbest = G;
% Calculate Neighborhood of each sample
%[NS,NS_num] = genarate_neighborhood(K_mu,tau);%% tau*num

% Ai_sum = zeros(sample_num);
% for i = 1:sample_num
%     Ai_sum(NS(1:NS_num(i),i),NS(1:NS_num(i),i)) = Ai_sum(NS(1:NS_num(i),i),NS(1:NS_num(i),i)) + 1;
% end
% Ai_sum = Ai_sum./sample_num;
% 
% Mi_sum = zeros(kernel_num);
% M = zeros(kernel_num);
% for p = 1:kernel_num
%     for q = 1:kernel_num
%         Mi_sum(p,q) = trace(Ai_sum.*K(:,:,p)*K(:,:,q));
%         M(p,q) = trace(K(:,:,p)'*K(:,:,q));
%     end
% end


t = 0;
flag = 1;

while flag
        %% update G with Z and K
    %fprintf('update mu with H and G in t: %d\n', t);
    %tic;
    %[mu] = update_mu(K,G,M,Mi_sum,rho,lambda);
    %toc;
    B = ((lambda*Z'*K_mu'+K_mu')/(1+lambda))';
    B = (B+B')/2;
    [V,D] = eig(B);
diagD = diag(D);
diagD(diagD<eps)=0;
G = V*diag(diagD)*V';
G = (G+G')/2;
%     [Ug,Sg,Vg] = svd(temp2,'econ');
%     G =Ug*Vg';
    %% update K with G and Z
    %fprintf('update G with K and H in t: %d\n', t);
    %tic;
    %G = update_G(K_mu,Ai_sum,rho, H);
    %toc;
%     coef = zeros(1,kernel_num);
%     for p=1:kernel_num
%         coef(1,p) = trace(K(:,:,p)*(lambda*Z*G'+G')); 
%     end
%     %gamma = coef/norm(coef,2);
%     gamma = coef/sum(coef);
M = zeros(kernel_num);
for p = 1:kernel_num
    for q = 1:kernel_num
        %Mi_sum(p,q) = trace(Ai_sum.*K(:,:,p)*K(:,:,q));
        M(p,q) = trace(K(:,:,p)'*K(:,:,q)*(eye(sample_num)+lambda*(Z*Z')));
    end
end
[mu] = update_mu(K,G,Z,M,lambda);
    K_mu = combine_K_mu(K,mu);
    %% update Z with K and G
    %fprintf('update H with G in t: %d\n', t);
    %tic;
    %H = my_kernel_kmeans(Ai_sum.*G,class_num);
%     temp1 = K_mu'*G;
%     [Uz,Sz,Vz] = svd(temp1,'econ');
%     Z = Uz*Vz';
    %toc;
    temp1 = (lambda/beta)*K_mu'*G;
    for i = 1:sample_num
        index = setdiff(1:sample_num,i);
        Z(i,index) = EProjSimplex_new(temp1(i,index) / 2);
        %Z(i,:) = EProjSimplex_new(temp1(i,:) / 2);
    end
%     %% update G with Z and K
%     %fprintf('update mu with H and G in t: %d\n', t);
%     %tic;
%     %[mu] = update_mu(K,G,M,Mi_sum,rho,lambda);
%     %toc;
%     B = ((lambda*Z'*K_mu'+K_mu')/(1+lambda))';
%     B = (B+B')/2;
%     [V,D] = eig(B);
% diagD = diag(D);
% diagD(diagD<eps)=0;
% G = V*diag(diagD)*V';
% G = (G+G')/2;
% %     [Ug,Sg,Vg] = svd(temp2,'econ');
% %     G =Ug*Vg';
%     %% update K with G and Z
%     %fprintf('update G with K and H in t: %d\n', t);
%     %tic;
%     %G = update_G(K_mu,Ai_sum,rho, H);
%     %toc;
% %     coef = zeros(1,kernel_num);
% %     for p=1:kernel_num
% %         coef(1,p) = trace(K(:,:,p)*(lambda*Z*G'+G')); 
% %     end
% %     %gamma = coef/norm(coef,2);
% %     gamma = coef/sum(coef);
% M = zeros(kernel_num);
% for p = 1:kernel_num
%     for q = 1:kernel_num
%         %Mi_sum(p,q) = trace(Ai_sum.*K(:,:,p)*K(:,:,q));
%         M(p,q) = trace(K(:,:,p)*(eye(sample_num)+lambda*(Z*Z'))*K(:,:,q)');
%     end
% end
% [mu] = update_mu(K,G,Z,M,lambda);
%     K_mu = combine_K_mu(K,mu);
    %%
    t = t+1;
    %fprintf('cal objective t: %d\n', t);
    %tic;
    %[obj(t)] = cal_obj(G,H,K_mu,Ai_sum,Mi_sum,mu,rho,lambda);
    obj(t) = norm(G-K_mu,'fro')^2+lambda*norm(G-K_mu*Z,'fro')^2+beta*norm(Z,'fro')^2;
    %obj(t) = norm(K_mu-G,'fro')^2;
    %toc;
    if (t>=2) && (abs((obj(t-1)-obj(t))/(obj(t-1)))<1e-20|| t>100)
    %if (t>=2) && (t>100)
    %if t>=2
        flag =0;
    end
end
K= (G+G')/2;
opt.disp = 0;
[H,~] = eigs(K,k,'LA',opt);
H_normalized = H./ repmat(sqrt(sum(H.^2, 2)), 1,k);
% plot(obj,'r','LineWidth',1.5);
end
