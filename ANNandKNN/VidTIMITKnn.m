load 'X_test.mat'
load 'Y_test.mat'

load 'X_Train.mat'
load 'Y_Train.mat'

VidTIMITModel = fitcknn(X_train,y_train,'NumNeighbors',7);
prediction =  predict(VidTIMITModel,X_test);

correctPredictions = prediction == y_test';
correctCount  = sum(correctPredictions);
accuracyPercentage = correctCount*100/numel(prediction)