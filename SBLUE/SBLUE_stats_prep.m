function SBLUEprep=SBLUE_stats_prep(covfunc,hypcov,Xtrain,xstar,q)
        % Compute the variables in SBLUE without knowlegde of the transition
        % matrix A
        %
        % Input : 
        % covfunc : the cov function of spatial field
        % hypcov : parameter of cov function 
        % q : the threshold of binary Spatial field
        % Xtrain: the location of existing sensors
        % xstar : the location of precidtion
        
        % Outpute : SBLUE prep
    
        c=norminv(1-q);
        N=size(Xtrain,1);
        K=feval(covfunc{:},hypcov,Xtrain); % Cov_Xtrain
        Cov_xstar=feval(covfunc{:}, hypcov, xstar, Xtrain); % Cov(xsatr, Xtrain)
 
        

        P1=zeros(N);P2=zeros(N);P3=zeros(N);P4=zeros(N);
        T1=[c c];
        T2=[c -c];
%         T3=[-c c];
%         T4=[-c -c];
        for i=1:N - 1
            for j=i+1:N
                K1=[1,K(i,j);K(i,j),1];
                K2=[1,-K(i,j);-K(i,j),1];
%                 if min(eig(Ktemp))<0 || ~issymmetric(Ktemp)
%                     disp(Ktemp)
%                     error("Ktemp is not psd")
%                 end
                P1(i,j)=binmcdf(T1,zeros(1,2),K1);
                P2(i,j)=binmcdf(T2,zeros(1,2),K2);
                P3(i,j)=P2(i,j); % Due to symmetry about y=x
                P4(i,j)=1-P1(i,j)-P2(i,j)-P3(i,j); % the probabilities sum to 1
            end
        end
        
        
        SBLUEprep.Cov_xstar=Cov_xstar;
        SBLUEprep.CovP=struct("P1",P1,"P2",P2,"P3",P3,"P4",P4);
end