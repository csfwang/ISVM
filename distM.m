function coef=distM(Y,doubletLabel,Xtrain)
p=doubletLabel(:,1);
q=doubletLabel(:,2);
Xm=(Xtrain(:,p)-Xtrain(:,q));

MXm=Xm.*(Y*Xm);
coef=sum(MXm)';
end