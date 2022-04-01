function saveFig(rootdir, format)
    % save .fig files in given directory in the specified format
    assert(isa(format,'char'),"error input")
    assert(format(1)=='.',"Enter a picture format")
    d = dir(fullfile(rootdir,'*.fig'));
    nfiles = numel(d);
    for i = 1 : nfiles
        figname = extractBetween(d(i).name, '','.');
        fig = openfig(d(i).name,'invisible');
        if strcmp(format,'.eps')
            set(gcf,'renderer','Painters')
            saveas(fig,strcat(figname{:}, format));   
            print('-depsc','-tiff','-r300',strcat(figname{:},'.eps'));
        else
            saveas(fig,strcat(figname{:}, format));  
        end
            
    end
end