function [pairlabel,pairs]=ConstructPair(X,Y,chit,cmiss)
% function
% [pairlabel,pairs]=ConstructPair(X,Y,chit,cmiss)
% Construct sample pair set based on training instances.
%
% Input:
%
% X = training samples (each column is a sample)
% Y = labels
% chit = the number of hits for each sample
% cmiss = the number of misses for each sample
%
% Output:
%
% pairlabel = the labels of each doublet (in row vector form)
% pairs = the indexes of doublet elements in the training set (X)
% (each row is a pair)
%

[dim,sampleNum]=size(X);
pairlabel=zeros(1,(chit+cmiss)*sampleNum);
pairs=zeros((chit+cmiss)*sampleNum,2);
indexzr=1;
for i=1:sampleNum
    HitDist=Inf*ones(sampleNum,1);
    MissDist=Inf*ones(sampleNum,1);
    Xik=X-X(:,i)*ones(1,sampleNum);
    Xik=Xik.^2;
    Distik=sum(Xik,1);
    SameLabel=find(Y==Y(i));
    DiffLabel=find(Y~=Y(i));
    HitDist(SameLabel)=Distik(SameLabel);
    MissDist(DiffLabel)=Distik(DiffLabel);
    HitDist(i)=Inf;
    [~,SortedHitIndex]=sort(HitDist);
    [~,SortedMissIndex]=sort(MissDist);
    HitSet=SortedHitIndex(1:chit);
    MissSet=SortedMissIndex(1:cmiss);
    for k=union(HitSet',MissSet')
        pairs(indexzr,:)=[i,k];
        if Y(i)==Y(k)
            pairlabel(indexzr)=-1;
        else
            pairlabel(indexzr)=1;
        end
        indexzr=indexzr+1;
    end
end
end