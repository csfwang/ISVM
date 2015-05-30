load('PenDigits.mat');
% Construct pairs of instances based on the nearest neighbor strategy
[pairlabel,pairs]=ConstructPair(Xtrain,Ytrain,1,1);
% Train the Mahalanobis metric using Doublet-SVM
[M,trainTime,dualgap]=PCML(Xtrain,pairlabel,pairs,1);
% Classify the test samples by kNN classifier (k=1)
correctRate=knnClassify(Xtrain,Xtest,Ytrain,Ytest,M,1);
disp(strcat('correct rate:',num2str(correctRate),'%, error:',num2str(100-correctRate),'%'));
disp(strcat('training time:',num2str(trainTime),'s'));