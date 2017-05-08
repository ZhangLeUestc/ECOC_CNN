function Y=ECOC_coding(nClass,nClassifier)
s = RandStream('mcg16807','Seed',0);
RandStream.setGlobalStream(s);
Y=rand(nClass,nClassifier);
Y(Y>=0.5)=1;
Y(Y<0.5)=0;
for i=1:nClass
    for j=1:nClass
        if j~=i
         if sum(abs(Y(i,:)-Y(j,:)))==0
             fprintf('Bad Coding Matrix\n');
         end
        end
        
    end
    
end







end