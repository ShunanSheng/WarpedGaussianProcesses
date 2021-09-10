function plotROC(TP,FP,FigTitle,FigLegend)
    %%% Plot the ROC curve given the TP vs FP
    if ~exist('FigLegend','var')||isempty(FigLegend)
        FigLegend="True Positive rate vs False Positive rate";
    end
    if ~exist('FigTitle','var')||isempty(FigTitle)
        FigTitle="Receiver operating characteristic curve";
    end
    
    figure();
    s=size(TP,2);
    if s>1
        for i=1:s
            fp=FP(:,i);
            tp=TP(:,i);
            plot(fp,tp,'DisplayName',FigLegend{i});
            hold on;
        end
    else
        FP=FP(:);
        TP=TP(:);
        plot(FP,TP,'DisplayName',FigLegend);
    end
    
   
    ylim([-0.1,1.2]);
    h=refline(1,0);
    h.LineStyle='--';
    h.Color='r';
    h.DisplayName="y=x";
    title(FigTitle);
    legend('-DynamicLegend');
    xlabel("False Positive rate")
    ylabel("True Positive rate")
end