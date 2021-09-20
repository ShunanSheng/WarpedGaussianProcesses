function SBLUE=SBLUE_stats(SBLUEprep,A1,A2,liP,liI,q)
    % Compute the mean, covariance in SBLUE with knowlegde of the transition matrix
    %
    % Input : 
    % SBLUEprep: the SBLUE statistics without knowing the transition matrix
    % A
    % A1 : transition matrix from WGPLRT
    % A2 : transition matirx from NLRT
    % liI : the vector with logical values tracking the location of integral observations
    % e.g. liI=(1,0,0,1,0,1)'
    % liP : the vector with logical values tracking the location of point observations
    % e.g. liP=ones(6,1)-liI
    % q : the threshold of binary Spatial field
    %
    % Output: SBLUE
    
    
    c=norminv(1-q);
    CovP=SBLUEprep.CovP;
    Cov_xstar=SBLUEprep.Cov_xstar;
    
    N=size(Cov_xstar,2);
    
    wp01=A1(3);wp11=A1(4);np01=A2(3);np11=A2(4); 
    % wp stands for the probability under WGPLRT
    % np stands for the probability under NLRT
    p=SBLUE_confusion;
    % compute the adjusted confusion probability
    
    % compute parameters
    mY=meanY(p,c,N);        
    Cov_Y=covY(p,CovP,mY,c);
    Cov_g=covgY(p,Cov_xstar,c);
    
    % create strtucture
    SBLUE.mY=mY;
    SBLUE.CovY=Cov_Y;
    SBLUE.Covg=Cov_g;
    SBLUE.c=c;
    
    function p=SBLUE_confusion
        p01=wp01*liP+np01*liI;
        p11=wp11*liP+np11*liI;
        
        p.p01=p01;
        p.p11=p11;
    end
 
end


%
% p.p1=p01*p01';
%         p.p2=p01*p11';
%         p.p3=p11*p01';
%         p.p4=p11*p11'; 