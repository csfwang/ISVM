function coef=distcoef(eta,doubletLabel,Xtrain)

p=doubletLabel(:,1);
q=doubletLabel(:,2);
Xm=Xtrain(:,p)-Xtrain(:,q);
M=Xm*(bsxfun(@times,Xm,eta'))';

MXm=Xm.*(M*Xm);
coef=sum(MXm)';
end