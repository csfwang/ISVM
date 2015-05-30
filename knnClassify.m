function correctRate=knnClassify(Xtrain,Xtest,Ytrain,Ytest,M,k)
    testSampleNum=size(Xtest,2);
    trainSampleNum=size(Xtrain,2);
    MXtrain=(M*Xtrain).*(Xtrain);
    sumxx=sum(MXtrain,1);
    sumxy=Xtrain'*M*Xtest;
    MXtest=(M*Xtest).*(Xtest);
    sumyy=sum(MXtest,1);
    dist=repmat(sumxx',1,testSampleNum)-2*sumxy+repmat(sumyy,trainSampleNum,1);
    if (k==1)
        [~, minindex]=min(dist);
        recoLabel=(Ytrain(minindex));
        correctNum=length(find(recoLabel==Ytest));
    else
        [~, minindex]=mink(dist,k);
        recoLabel=mode(Ytrain(minindex));
        correctNum=length(find(recoLabel'==Ytest));
    end
    correctRate=correctNum/testSampleNum*100;
    disp(num2str(correctRate));
end