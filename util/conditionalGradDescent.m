function [hhat,vhat,A]=conditionalGradDescent()
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

    
end