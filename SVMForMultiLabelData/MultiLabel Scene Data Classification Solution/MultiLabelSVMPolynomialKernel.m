load X_train.mat
load X_test.mat
load Y_train.mat
load Y_test.mat

YFullMat=y_train;
YFullMat(YFullMat==0)=-1;

SvmTrainModel = cell(6,1);
predictY=cell(6,1);
Matrix=zeros(6,size(y_test,1));


for k=1:6
    SvmTrainModel{k} = fitcsvm(X_train,YFullMat(:,k),'KernelFunction','polynomial','PolynomialOrder',2,'BoxConstraint',10000,'KernelScale','auto');
end

for k=1:6
    predictY{k} = predict(SvmTrainModel{k},X_test);
end

for k=1:6
	Matrix(k,:)=predictY{k};
	Matrix(Matrix==-1)=0;
end

AndMatrix=y_test&Matrix'; 
OrMatrix=y_test|Matrix';
sumOfAnds=sum(AndMatrix,2);
sumOfOrs=sum(OrMatrix,2);
ArrayOfJaccards=zeros(size(y_test,1),1);

for i=1:numel(sumOfAnds)
ArrayOfJaccards(i,:)=sumOfAnds(i,:)/sumOfOrs(i,:);
end

AccuracyPercentage=sum(ArrayOfJaccards)/numel(sumOfAnds)*100