function [M,trainTime,dualgap]=NCML(Xtrain,pairlabel,pairs,C)
% The NCML training function
% Xtrain: training set (dxN, d is sample dimension and N is sample number)
% pairlabel: the label of pairs (-1 or 1)
% pairs: the index of pairs, each column is a pair, the first and
% second rows are the indexes of the first sample and second sample in
% Xtrain.
% C: the coefficient of the loss term
% Return:
% M: the learned distance metric
% trainTime: the training time (sec)
% dualitygap: the duality gap of each iteration
t1=clock;
maxiter=100;
kernelcache=Xtrain'*Xtrain;
l=size(pairs,1);
eta=ones(l,1)*0;
miu=ones(l,1)*(-Inf);
dualgap=zeros(1,maxiter);
for iter=1:maxiter
    coef5=distcoef(eta,pairs,Xtrain);
    coef1=1-coef5.*double(pairlabel');%delta
    beta=svmtrain(int32(pairlabel'),int32(pairs),kernelcache,-coef1,['-s 0 -t 5 -d 2 -f 1 -c ' num2str(C) ' -q 1']);
    fprintf(strcat('Iteration ',num2str(iter),'|'));
    coef2=distcoef(double(pairlabel').*beta,pairs,Xtrain);%gamma
    if max(coef2(:))<0
        miu=zeros(l,1);
    else
        miu=svmtrain(int32(pairlabel'),int32(pairs),kernelcache,-coef2,'-s 3 -t 5 -d 2 -f 1 -c 10000 -q 1');
    end
    eta=miu-double(pairlabel').*beta;
    coef3=distcoef(miu,pairs,Xtrain);
    dualgap(iter)=DualGapNCML(pairlabel',coef3,beta,eta,coef2,C);
    fprintf(strcat('dualgap=',num2str(dualgap(iter)),'|'));
    if iter==1
        maxdualgap=dualgap(iter);
    end
        if dualgap(iter)<max(maxdualgap*0.01,0.1) && iter>2
            break;
        end
        disp('.');
end
dualgap=dualgap(1:iter);
p=pairs(:,1);
q=pairs(:,2);
Xm=Xtrain(:,p)-Xtrain(:,q);
M=Xm*(bsxfun(@times,Xm,miu'))';

t2=clock;
trainTime=etime(t2,t1);
end