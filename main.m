%_________________________________________________________________________%
%  Extreme learning machine based classiier for diagnosis
% of COVID-19 using deep convolutional network (E-DiCoNet) source codes demo V1.0        %
%                                                                         %
%  Developed in MATLAB R2018b                                             %
%                                                                         %
%  Author and programmer: Tripti Goel                                     %
%                                                                         %
%         e-Mail: triptigoel83@gmail.com                                  %
%                 triptigoel@ece.nits.ac.in                               %
%                                                                         %
%       Homepage: http://www.nits.ac.in/departments/ece/ece.php           %
%                                                                         %
%  Main paper:  R Murugan and Tripti Goel                                 %
%               E-DiCoNet: Extreme learning machine based classiier for 
%               diagnosis of COVID-19 using deep convolutional network %
%               Journal of Ambient Intelligence and Humanized Computing ,              
%               DOI: https://doi.org/10.1007/s12652-020-02688-3
   %
%                                                                         %
%_________________________________________________________________________%

digitDatasetPath = fullfile('C:\Users\MIPLAB\Documents\Covid\XRay\MultiObjectiveGoA\NewX-rayDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'FileExtensions',{'.jpg','.png','.jfif','.jpeg'},'LabelSource','foldernames');

tbl = countEachLabel(imds);

minSetCount = min(tbl{:,2});

net = resnet50();

no_person = 3;

    [trainingSet, testSet] = splitEachLabel(imds, 0.7);
    
    tStart = tic; 
       
    imageSize = net.Layers(1).InputSize;
    augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet,'ColorPreprocessing', 'gray2rgb');
    augmentedTestSet = augmentedImageDatastore(imageSize, testSet,'ColorPreprocessing', 'gray2rgb');
    
    
    featureLayer = 'fc1000';
    trainingFeatures = activations(net, augmentedTrainingSet, featureLayer, ...
        'MiniBatchSize', 64, 'OutputAs', 'columns');
    
    trainingFeatures = double(trainingFeatures);
    testLabels = testSet.Labels;
    % Get training labels from the trainingSet
    trainingLabels = trainingSet.Labels;
    
    testFeatures = activations(net, augmentedTestSet, featureLayer, ...
        'MiniBatchSize', 64, 'OutputAs', 'columns');
    
    testFeatures = double(testFeatures);
    fprintf('Creating the target matrix for tarining of ANN\n');
    
    dimTrain = size(trainingSet.Files,1);
    dimTest = size(testSet.Files,1);
    
    no_img_p_s_train = 630;
    TrainTargets = zeros(no_person, dimTrain);
    for j = 1:no_person
        for k = 1:no_img_p_s_train
            TrainTargets(j,((j-1)*no_img_p_s_train + k)) = 1;
        end
    end
    fprintf('Saving on disk TrainTargets \n'); save TrainTargets  TrainTargets ;
    % %
    fprintf('Creating the target matrix of TestData for calculating accuracy(Performance)\n');
    no_img_p_s_test = 900-no_img_p_s_train;
    
    TestTargets = zeros(no_person, dimTest);
    for j = 1:no_person
        for k = 1:no_img_p_s_test
            TestTargets(j,((j-1)*no_img_p_s_test + k)) = 1;
        end
    end
    fprintf('Saving on disk TestTargets \n'); save TestTargets TestTargets;
    
    TrainTargets_ind=vec2ind(TrainTargets);
    TestTargets_ind=vec2ind(TestTargets);
    
    ELM_Train=[TrainTargets_ind' trainingFeatures'];
    ELM_Test=[TestTargets_ind' testFeatures'];
    
    save ELM_Train ELM_Train
    save ELM_Test ELM_Test
    
    test=zeros(50,1);
    train=zeros(50,1);
    train_time=zeros(50,1);
    testing_time=zeros(50,1);
    wb=waitbar(0,'Please waiting...');
    
    for rnd = 1 : 50
    waitbar(rnd/50,wb);
    
    [TrT, TtT, TrainAcc, TestAcc] = My_ELM_Old(ELM_Train, ELM_Test,1, 20000, 'sig');
    
    test(rnd,1)=TestAcc;
    train(rnd,1)=TrainAcc;
    train_time(rnd,1)=TrT;
    testing_time(rnd,1)=TtT;
    
    end
   close(wb);
tEnd = toc(tStart)      

AverageTrainingTime=mean(train_time)
StandardDeviationofTrainingTime=std(train_time)
AvergeTestingTime=mean(testing_time)
StandardDeviationofTestingTime=std(testing_time)
AverageTrainingAccuracy=mean(train)
StandardDeviationofTrainingAccuracy=std(train)
AverageTestingAccuracy=mean(test)
StandardDeviationofTestingAccuracy=std(test)
    
    load PredictedLabels; load ExpectedLabel; load TY;
     
    confMat = confusionmat(ExpectedLabel, PredictedLabels);

    confMat1 = bsxfun(@rdivide,confMat,sum(confMat,2));

  EVAL = Evaluate(ExpectedLabel, PredictedLabels)
  
  plotConfMat(confMat, {'COVID', 'Normal', 'Pneumonia'});
  
  plotroc(TestTargets, TY)
   
   
   