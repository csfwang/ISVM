function [dualgap,b]=DualGapPCML(pairlabel,pairs,Xtrain,lambda,Y,C)
%Return the duality gap of each iteration of PCML

pairlabel=pairlabel';
p=pairs(:,1);
q=pairs(:,2);
Xm=Xtrain(:,p)-Xtrain(:,q);
M=Xm*(bsxfun(@times,Xm,(lambda.*pairlabel)'))';

M=M+Y;

idx=find(lambda>0&lambda<C);

MXm=Xm.*(M*Xm);
dist=sum(MXm)';

if isempty(idx)
    fprintf('idx is empty!');
    b=selectxib(pairlabel,dist);
else
    bsum=1./pairlabel(idx)-dist(idx);
    b=sum(bsum(:))/length(idx);
end

xi=1-pairlabel.*(dist+b);

xi=max(xi,0);
dualgap=norm(M,'fro')^2;
dualgap=dualgap+C*sum(xi(:))-sum(lambda(:));
end