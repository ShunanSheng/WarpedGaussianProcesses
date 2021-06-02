+%%% Quadratric approximation
% Function Q(v)=-1/2*G(v)^T K_N^{-1} G(v)+ sum log dig(Jacobian)

global G;
G=@(Z,a,b) norminv(betacdf(Z,a,b));
cov=@covFunc;
mean=@meanFunc;
global a;global b; a=10;b=5;
N=10;l0=0.5;l1=3;sigma2=0.01;

X=linspace(-8,8,N);
Cov0=feval(cov,X,X,l0);
Cov0_chol=chol(Cov0+1e-9*eye(N)); 
Cov0_ichol=(Cov0_chol\eye(N))'; 
Cov0_logdet=2*trace(log(Cov0_chol));

Cov1=feval(cov,X,X,l1);
Cov1_chol=chol(Cov1+1e-9*eye(N)); 
Cov1_ichol=(Cov1_chol\eye(N))'; 
Cov1_logdet=2*trace(log(Cov1_chol));


[v0,A0]=FW(Cov0_ichol);
A0_inv=inv(A0);

A0_chols=chol(A0_inv+sigma2*eye(N));
A0_ichols=(A0_chols\eye(N))';

[v1,A1]=FW(Cov1_ichol);
A1_chol=chol(A1+1e-9*eye(N));
A1_ichol=(A1_chol\eye(N))';
A1_inv=A1_ichol'*A1_ichol;

A1_chols=chol(A1_inv+sigma2*eye(N));
A1_ichols=(A1_chols\eye(N))';


%% True Nloglik

% 
% 
K=1000;
ListC=-0.5:0.01:0.5;
alphas=[0]*length(ListC);powers=[0]*length(ListC);
m=mean(X);
j=1;
for c=ListC
    decision=[0]*K;ground_truth=[0]*K;
    for i=1:K
        % sample from independent beta process
        if rand<0.5
            ground_truth(i)=0;
            Y=Cov0_chol*randn(N,1)+m;
            U=normcdf(Y);
            Z=betainv(U,a,b)+sqrt(sigma2)*randn(N,1);
        else
            ground_truth(i)=1;
            Y=Cov1_chol*randn(N,1)+m;
            U=normcdf(Y);
            Z=betainv(U,a,b)+sqrt(sigma2)*randn(N,1);
        end
        nloglik0=NlogLik(Z,v0,A0_chols,A0_ichols,Cov0_chol,Cov0_ichol);
        nloglik1=NlogLik(Z,v1,A1_chols,A1_ichols,Cov1_chol,Cov1_ichol);
        if (nloglik0-nloglik1)>2*c
            decision(i)=1;
        else
            decision(i)=0;
        end
    end
    diff=decision-ground_truth;
    alphas(j)=sum(diff==1)/ sum(ground_truth == 0);
    powers(j)=1-sum(diff==-1)/ sum(ground_truth == 1);
    j=j+1
end
figure;
scatter(alphas,powers,1,'r');
title("ROC curve: Monte Carlo");
xlabel("False alarm");
ylabel("Power");
hline = refline([1 0]);
hline.Color = 'b';


%% Functions

function nloglik=NlogLik(v,v_hat,A_chol,A_ichol,cov_chol,cov_ichol)
    diff_v=A_ichol*(v-v_hat);
    nloglik=trace(log(A_chol))+trace(log(cov_chol))-Q(v_hat,cov_ichol)+diff_v'*diff_v/2;  
end

function dG=gradient_G(v)
    global a;global b;global G;
    Gv=G(v,a,b);
    dg=betapdf(v,a,b)./normpdf(Gv);
    dG=diag(dg);
end

%check
function hG=hessian_G(v)
    global a;global b;global G;    
    N=length(v);
    Gv=G(v,a,b);
    hG=(dbeta(v,a,b).*normpdf(Gv).^2-dnorm(Gv).*betapdf(v,a,b).^2)./(normpdf(Gv).^3);
    hG=diag(hG);
end

% derivative of beta pdf
function dh=dbeta(v,a,b)
    global G;
    BetaV=betapdf(v,a,b);
%     (betapdf(0.5+delta,10,5)-betapdf(0.5,10,5))/delta
    dh=(a-1)*BetaV./v-(b-1)*BetaV./(1-v);
end

% derivative of norm(0,1) pdf
function df=dnorm(v)
    df=normpdf(v).*(-v);
end
% gradient of Q(v)
function dQ=gradient_Q(v,Cov_ichol)
    global a;global b;global G;    
    N=length(v);
    dG=gradient_G(v);
    Gv=G(v,a,b);
    dQ=-Gv'*Cov_ichol'*Cov_ichol*dG;
    dJacobian=(dbeta(v,a,b).*normpdf(Gv).^2-dnorm(Gv).*betapdf(v,a,b).^2)./(normpdf(Gv).^2.*betapdf(v,a,b));
    dQ=dQ'+dJacobian;
end

% hessian of Q(v)
function hQ=hessian_Q(v,Cov_ichol)
     global G;global a;global b;
     N=length(v);
     Gv=G(v,a,b);
     dG=betapdf(v,a,b)./normpdf(Gv);
     hG=diag((dbeta(v,a,b).*normpdf(Gv).^2-dnorm(Gv).*betapdf(v,a,b).^2)./(normpdf(Gv).^3));
     Cov_inv=Cov_ichol'*Cov_ichol;
     
     hQ=-diag(Gv'*Cov_inv)*hG-Cov_inv.*(dG*dG');
     
     hJacobian=zeros(N,1);
     dJacobian=(dbeta(v,a,b).*normpdf(Gv).^2-dnorm(Gv).*betapdf(v,a,b).^2)./(normpdf(Gv).^2.*betapdf(v,a,b));
     delta=0.000001;
     v_diff=v+delta*ones(N,1);
     Gv_diff=G(v_diff,a,b);
     dJacobian_diff=(dbeta(v_diff,a,b).*normpdf(Gv_diff).^2-dnorm(Gv_diff).*betapdf(v_diff,a,b).^2)./(normpdf(Gv_diff).^2.*betapdf(v_diff,a,b));
     
     for i=1:N
        hJacobian(i)= (dJacobian_diff(i)-dJacobian(i))/delta;     
     end
     hQ=hQ+diag(hJacobian);
end


function Qv=Q(v,Cov_ichol)
    global G;global a;global b;
    Gv=G(v,a,b);
    Gv_chol=Cov_ichol*Gv;
    Qv=-1/2*Gv_chol'*Gv_chol+sum(log(betapdf(v,a,b)./normpdf(Gv)));   
end


function K=covFunc(X,Y,l)
    D=pdist2(X',Y');
    K=exp(-D.^2/2/l.^2);
end

function m=meanFunc(X)
    n=length(X);
    m=zeros(n,1);
end

% The major problem occurs in minimizing the objective function

function [v_hat,A]=FW(Cov_ichol)

    % use Frank-Wolfe method to calculate find the local minimum for -Q(v)
    % the constraint is 0<=v_i<=1
    N=size(Cov_ichol,1);
    error=0.01;
    % what shall be the boundary
    lb=zeros(N,1);
    ub=ones(N,1);
    k=1;max_iter=100;
    V=linspace(0.001,0.999,N)';
    % to trace the function value and the difference
    Q_val=[0]*max_iter;
    Diff=[0]*max_iter;
    
    % main loop
    while k<max_iter
        f=-gradient_Q(V,Cov_ichol);
        v_bar=linprog(f,[],[],[],[],lb,ub);
        Diff(k)=f'*(v_bar-V);
        if abs(f'*(v_bar-V))<error
            break
        end
        Qline=@(lambda) -Q(V+lambda*(v_bar-V),Cov_ichol);
        lambda=fminbnd(Qline,0,1);
        V=V+lambda*(v_bar-V);
        Q_val(k)=Q(V,Cov_ichol);
        k=k+1;
    end
    v_hat=V;
    A=-hessian_Q(v_hat,Cov_ichol);

% it does not seem to converge
%     figure;
%     plot((1:1:length(Q_val)),Q_val)
%     figure
%     plot((1:1:length(Diff)),Diff)
end




%% Auxiliary

%% Check hessian and gradient
% v=linspace(0.001,0.999,N)';
% delta=0.000001;
% 
% 
% hQ=hessian_Q(v,Cov1_ichol)
% 
% 
% % Gradient
% dQ=gradient_Q(v,Cov1_ichol);
% Qv=Q(v,Cov1_ichol);
% dQ_num=zeros(N,1);
% for i=1:N
%     d=zeros(N,1);d(i)=delta;
%     Qv_diff=Q(v+d,Cov1_ichol);
%     dQ_num(i)=(Qv_diff-Qv)/delta;
% end
% dQ_num;
% 
% % Hessian
% hQ_num=zeros(N);
% 
% for i=1:N
%     d=zeros(N,1);d(i)=delta;
%     dQ_diff=gradient_Q(v+d,Cov1_ichol);
%     for j=1:N
%         hQ_num(i,j)=(dQ_diff(j)-dQ(j))/delta;
%     end
% end
% hQ_num


%% Plot of Z: Verify the histogram of Z follows beta(10,5) when there is no dependency
% N=10000;l1=0.0001;
% X=linspace(-8,8,N);
% m=mean(X);
% Cov=cov(X,X,l1);
% Cov_chol=chol(Cov);
% Y=Cov_chol*randn(N,1)+m;
% U=normcdf(Y);
% Z=betainv(U,a,b);
% 
% 
% figure;
% histogram(Y,100,'Normalization',"probability")
% figure;
% histogram(Z,100,'Normalization',"probability")








