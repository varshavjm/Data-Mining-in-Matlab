%LOAD TRAINING AND TEST DATA
load 'X_test.mat'
load 'Y_test.mat'
load 'X_Train.mat'
load 'Y_Train.mat'

%CONVERT DATA INTO FORMAT NEEDED BY TRAIN FUNCTION
y_trainVector = ind2vec(y_train);
y_testVector = ind2vec(y_test);
y_trainVectorFull = full(y_trainVector);
y_testVectorFull = full(y_testVector);

%USE FEED FORWARD NEURAL NETWORK FOR TRAINING
netlm = feedforwardnet(25,'trainlm');
netlm = train(netlm,X_train',y_trainVectorFull);
%netlm = train(netlm,X_train',y_trainVectorFull,'useParallel','yes','useGPU','yes','showResources','yes')

%PREDICT LABEL FOR TEST SAMPLE
predictionlm = netlm(X_test');
roundedPrediction = round(predictionlm);
correctlm = roundedPrediction == y_testVectorFull;
exactMatch = all(correctlm);

%CALCULATE ACCURACY
accuracyPercentage = sum(exactMatch)*100/numel(exactMatch)

---------------------------------------------------------------------------------------------------------------------------------------------
