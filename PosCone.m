function posM=PosCone(M)
M=(M+M')/2;
[V D]=eig(M);
D=max(D,1e-8);
posM=V*D*V';
end