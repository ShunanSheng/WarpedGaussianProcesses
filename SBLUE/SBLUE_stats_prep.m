function SBLUEprep=SBLUE_stats_prep(covfunc,meanfunc,hyp,Xtrain,xstar)
        % Compute the variables in SBLUE without knowlegde of the transition
        % matrix A
        %
        % Input : 
        % covfunc : the cov function of spatial field
        % hypcov : parameters
        % Xtrain: the location of existing sensors
        % xstar : the location of precidtion
        
        % Outpute : SBLUE prep
    
        N=size(Xtrain,1);
        mu=meanfunc(hyp.mean, Xtrain);
        K=feval(covfunc{:},hyp.cov,Xtrain); % Cov_Xtrain
        mXstar=meanfunc(hyp.mean,xstar);
        Cov_xstar=feval(covfunc{:}, hyp.cov, xstar, Xtrain); % Cov(xsatr, Xtrain)
        
        c=hyp.thres;
        
        P1=zeros(N);P2=zeros(N);P3=zeros(N);P4=zeros(N);
        T1=[c c];
        T2=[c -c];
        T3=[-c c];

        for i=1:N - 1
            for j=i+1:N
                mu1=[mu(i),mu(j)];
                mu2=[mu(i),-mu(j)];
                mu3=[-mu(i),mu(j)];
                
                K1=[K(i,i),K(i,j);K(i,j),K(j,j)];
                K2=[K(i,i),-K(i,j);-K(i,j),K(j,j)];
                K3=K2;
                
                P1(i,j)=binmcdf(T1,mu1,K1);
                P2(i,j)=binmcdf(T2,mu2,K2);
                P3(i,j)=binmcdf(T3,mu3,K3);
                P4(i,j)=1-P1(i,j)-P2(i,j)-P3(i,j); % the probabilities sum to 1
            end
        end
        
        SBLUEprep.mXstar=mXstar;
        SBLUEprep.chat=(c-mu)./diag(K);
        SBLUEprep.Cov_xstar=Cov_xstar;
        SBLUEprep.CovP=struct("P1",P1,"P2",P2,"P3",P3,"P4",P4);
end