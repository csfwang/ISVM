function dualgap=DualGapNCML(h,dist,beta,eta,gamma,C)
% Return the duality gap of each iteration of NCML
alpha=beta.*h+eta;

idx=find(beta>0&beta<C);
if isempty(idx)
    fprintf('idx is empty!');
    b=selectxib(h,dist);
else
    bsum=1./h(idx)-dist(idx);
    b=sum(bsum(:))/length(idx);
end

xi=1-h.*(dist+b);

xi=max(xi,0);
dualgap=C*sum(xi(:))-sum(beta(:))+alpha'*gamma;
end