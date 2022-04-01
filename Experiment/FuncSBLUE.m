function FuncSBLUE(TP,FP,)



L=length(FPR);
MSE=zeros(L,1);

M = 1000;
YT = repmat(Ytest,[1,M]);
for i = 1 : L 
    tp = TPR(i);
    fp = FPR(i); % Suppose ROC : y=x 
    A=[1-fp,fp;1-tp,tp]; % Define the confusion matrix for both A1,A2
    % Simulate the noisy data
    Yhat_noise=repmat( Ytrain, [1,M] );
    for j=1:M
        rnd = rand(length(Ytrain),1);
        rnd1 = rnd > tp; % filp the 1 to 0 with prob 1-tp
        rnd2 = rnd > (1-fp);% filp the 0 to 1 with prob fp
 
        Y1=Yhat_noise(:,j)==1;
        Y0=Yhat_noise(:,j)==0;
        
        idP1=logical(liP.*Y1);
        idP0=logical(liP.*Y0);
        
        idI1=logical(liI.*Y1);
        idI0=logical(liI.*Y0);
        
        Yhat_noise(idP1,j)=(1-rnd1(idP1)).*Yhat_noise(idP1,j)...
                                    +rnd1(idP1).*(1-Yhat_noise(idP1,j));
        Yhat_noise(idP0,j)=(1-rnd2(idP0)).*Yhat_noise(idP0,j)...
                                    +rnd2(idP0).*(1-Yhat_noise(idP0,j));
                                    
        Yhat_noise(idI1,j)=(1-rnd1(idI1)).*Yhat_noise(idI1,j)...
                                    +rnd1(idI1).*(1-Yhat_noise(idI1,j));
        Yhat_noise(idI0,j)=(1-rnd2(idI0)).*Yhat_noise(idI0,j)...
                                    +rnd2(idI0).*(1-Yhat_noise(idI0,j)); 
    end
    
    transitionMat=SBLUE_confusion(A,A,liP,liI);
    % Apply SBLUE
    SBLUE=SBLUE_stats(SBLUEprep,transitionMat,c);
    Ypred=SBLUE_pred(SBLUE,Yhat_noise);
    % Evaluate the MSE and Accuracy
    MSE(i)=sum((Ypred(:)-YT(:)).^2)/length(Ypred(:));
    if mod(i,floor(L/10))==0
        fprintf("Iteration %d, fp=%4.2f, MSE=%4.2f\n",i,fp, MSE(i));
    end
end



end