load X_train.mat
load X_test.mat
load Y_train.mat
load Y_test.mat

YDash=y_train';

YFullMat=zeros(3500,25);
YFullMat(YFullMat==0)=-1;

%Convert y train into required format
for k=1:3500
    YFullMat(k,YDash(k))=1;
end

SvmTrainModel = cell(25,1);
predictY=cell(25,1);

for k=1:25
    SvmTrainModel{k} = fitcsvm(X_train,YFullMat(:,k),'KernelFunction','polynomial','PolynomialOrder',2,'BoxConstraint',10000);
end

for k=1:25
    predictY{k} = predict(SvmTrainModel{k},X_test);
end

Matrix=zeros(25,1000);
for k=1:25
    Matrix(k,:)=predictY{k};
    Matrix(Matrix==-1)=0;
end

predictedAnswer=vec2ind(Matrix);

%Compare the predictedAnswer with predictLabel
correctPredictions=predictedAnswer==y_test;
correctCount=sum(correctPredictions);
accuracyPercentage=correctCount*100/numel(predictedAnswer)