function p=SBLUE_confusion(A1,A2,liP,liI)
    % Create a sample transition matrix when the sensors of the same type have the 
    % same type1,type2 error. 
    wp01=A1(3);wp11=A1(4);np01=A2(3);np11=A2(4); 
    % wp stands for the probability under WGPLRT
    % np stands for the probability under NLRT
    p01=wp01*liP+np01*liI;
    p11=wp11*liP+np11*liI;

    p.p01=p01;
    p.p11=p11;
end