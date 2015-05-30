function [M,trainTime,dualitygap]=PCML(Xtrain,pairlabel,pairs,C)
% The PCML training function
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
kernelCache=Xtrain'*Xtrain;
[dim,sampleNum]=size(Xtrain);
Y=zeros(dim);
dualitygap=zeros(1,maxiter);

for iter=1:maxiter
    fprintf(strcat('Iteration ',num2str(iter),'|'));
    dis=distM(Y,pairs,Xtrain);
    gam=1-(pairlabel').*dis;
    lambda=svmtrain(int32(pairlabel'),int32(pairs),kernelCache,-gam,['-s 0 -t 5 -d 2 -f 1 -c ' num2str(C) ' -q 1']);
    
    Xt=Xtrain(:,pairs(:,1))-Xtrain(:,pairs(:,2));
    Y0=-bsxfun(@times,lambda'.*pairlabel,Xt)*Xt';
    
    Y=PosCone(Y0);
    
    [dualitygap(iter),b]=DualGapPCML(pairlabel,pairs,Xtrain,lambda,Y,C);
    
    fprintf(strcat('dualgap=',num2str(dualitygap(iter)),'|b=',num2str(b),'|'));
    if iter==1
        maxdualgap=dualitygap(iter);
    end
        if dualitygap(iter)<max(maxdualgap*0.01,0.1) && iter>2
            dualitygap=dualitygap(1:iter);
            break;
        end
        disp('.');
end
Xt=Xtrain(:,pairs(:,1))-Xtrain(:,pairs(:,2));
M=bsxfun(@times,lambda'.*pairlabel,Xt)*Xt';

M=M+Y;

t2=clock;
trainTime=etime(t2,t1);
end