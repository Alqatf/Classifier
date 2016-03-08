function kNNOpt = kNN(fea_train,gnd_train)
kNNOpt = [];
kVal = 1;
for i = 1:16
%    
    mdl = fitcknn(fea_train, gnd_train, 'NumNeighbors',kVal,'kFold',5);
    %cvmodel = crossval(mdl,'kfold',5);
    cvError = kfoldLoss(mdl);
    cvCorr = 1 - cvError;
    kNNOpt = [kNNOpt; [kVal cvCorr]];

    kVal = kVal + 2;    
%{
    Fold_Number = 5;
    indices = crossvalind('Kfold',gnd_train, Fold_Number);
    cp = classperf(gnd_train);
    for j = 1:Fold_Number
        test = (indices == j);
        train = (indices ~= j);   % The rest part as train case
        mdl = knnclassify(fea_train(test,:),fea_train(train,:),gnd_train(train,:),kVal);
        classperf(cp,mdl,test);
    end;
        corr = cp.CorrectRate;

        kNNOpt = [kNNOpt; [kVal corr]];
        kVal = kVal + 2;
%}
end;
%#plot accuracy curves;
plot(kNNOpt(:,1),kNNOpt(:,2));
set(gca,'XTick',1:2:33);
title('kNN Classification');
xlabel('k value');
ylabel('classification accuracy');
saveas(gcf,'kNN.png');