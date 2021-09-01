function plotROC(TP,FP)
    %%% Plot the ROC curve given the TP vs FP
    figure();
    plot(FP,TP);
    ylim([-0.1,1.2]);
    h=refline(1,0);
    h.LineStyle='--';
    h.Color='r';
    title("Receiver operating characteristic curve");
    legend("True Positive rate vs False Positive rate","y=x");
    xlabel("False Positive rate")
    ylabel("True Positive rate")
end