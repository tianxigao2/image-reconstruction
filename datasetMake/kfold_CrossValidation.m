% In k-fold cross-validation, the original sample is randomly partitioned into k equal sized subsamples.
% Of the k subsamples, a single subsample is retained as the validation data for testing the model,
% and the remaining k ? 1 subsamples are used as training data.
% The cross-validation process is then repeated k times,
% with each of the k subsamples used exactly once as the validation data.
% The k results can then be averaged to produce a single estimation.
% The advantage of this method over repeated random sub-sampling (see below) is that
% all observations are used for both training and validation,
% and each observation is used for validation exactly once.
clear;
clc;

filename = ["cmp30.csv", "cmp50.csv", "cmp75.csv", "uncmp.csv"];
opTrainingArray = ["traing1.csv", "traing2.csv", "traing3.csv", "traing4.csv"];

csv11 = csvread(filename(1));
csv12 = csvread(filename(2));
csv13 = csvread(filename(3));
trainingSet1 = [csv11; csv12; csv13];
%csvwrite(opTrainingArray(i), trainingSet);
disp('run to the first iteration')

csv14 = csvread(filename(4));
totalTestingSet = [ csv14; csv11; csv12; csv13];
csvwrite('totalTestingSet.csv', totalTestingSet);
disp('finish the overall testing dataset')

csv21 = csvread(filename(2));
csv22 = csvread(filename(3));
csv23 = csvread(filename(4));
trainingSet2 = [csv21; csv22; csv23];
disp('run to the second iteration')

csv31 = csvread(filename(3));
csv32 = csvread(filename(4));
csv33 = csvread(filename(1));
trainingSet3 = [csv31; csv32; csv33];
disp('run to the third iteration')

csv41 = csvread(filename(4));
csv42 = csvread(filename(1));
csv43 = csvread(filename(2));
trainingSet4 = [csv41; csv42; csv43];
disp('almost done!!!!')

% c1 = csvread(opTrainingArray(1));
% c2 = csvread(opTrainingArray(2));
% c3 = csvread(opTrainingArray(3));
% c4 = csvread(opTrainingArray(4));
totalTrainingSet = [trainingSet1; trainingSet2; trainingSet3, trainingSet4];
csvwrite('totalTrainingSet.csv', totalTrainingSet);




    