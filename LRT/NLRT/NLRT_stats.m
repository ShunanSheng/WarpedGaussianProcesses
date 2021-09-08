function [D0,D1]=NLRT_stats(ZI,ZI0,ZI1,sumstats,d)
    % Conduct Neighbourhood density based LRT given the integral observations 
    %
    % Input: 
    % ZI   : the integral observations
    % ZI0, ZI1: the samples from null/alternative hypotheses
    % sumstats : summary statistics 
    % d    : distance mertic 
    %
    % Output: 
    % D0, D1: distance between ZI0,ZI1 and the integral observation ZI (nI)
    
    nI=size(ZI,2);J=size(ZI0,2);
    D0=zeros(J,nI);D1=zeros(J,nI);
    
    S=sumstats(ZI); % Compute the summary statistics of ZI, ZI0, ZI1
    S0=sumstats(ZI0); 
    S1=sumstats(ZI1);
    
    for i=1:nI
        y=S(:,i); % compute the summary statistics of the given observation
        D0(:,i)= d(S0,y)'; % compute the distance between S0 and the integral observation
        D1(:,i)= d(S1,y)';
    end
    
end
