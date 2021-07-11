function plotROC(TP,FP)
    %%% Plot the ROC curve given the TP vs FP
    figure();
    plot(FP,TP);
    h=refline(1,0);
    h.LineStyle='--';
    h.Color='r';
    title("Receiver operating characteristic curve");
    legend("True Positive rate","False Positive rate");
    xlabel("False Positive rate")
    ylabel("True Positive rate")
end