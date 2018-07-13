function[] = run(datadir_images, datadir_rects)
clc;

%% Loading in the data
% Load all images and extract the integral image as well
addpath('mp6code');

files = dir(strcat(datadir_images, '/', '*.jpeg'));

training_actual_im = zeros(480,720,126);
testing_actual_im = zeros(480,720,42);

training_imdata = zeros(480,720,126);
testing_imdata = zeros(480,720,42);

%Get integral and actual images of training data
for i = 1:126
    curr_im = imread(strcat(datadir_images,'/',files(i).name));
    training_actual_im(:,:,i) = sum(curr_im,3);
    training_imdata(:,:,i) = integralimage(curr_im);
end

% Get integral and actual images of testing data
for i = 1:42
    curr_im = imread(strcat(datadir_images,'/',files(i+126).name));
    testing_actual_im(:,:,i) = sum(curr_im,3);
    testing_imdata(:,:,i) = integralimage(curr_im);
end

%% Read the rectangles data from the text files and store into rects
%rects = readtable(strcat(datadir_rects,'/','allrects.txt'));
%rects = table2array(rects);
load(strcat(datadir_rects,'/','allrects.txt'),'-ascii')
face_rects = allrects(:,17:32);
non_face_rects = allrects(:,33:48);

%% Training utilizing the 126 Integral Images

W = ones(126,8);

%Renormalize weights
W = W/sum(sum(W));

%Binary Class Labels
Y = [ones(126,4) zeros(126,4)];

%classifiers
classifiers = zeros(40,8);

%weighted error rates
beta = zeros(40,1);

%adaboost weights
alpha = zeros(40,1);

%errors for each 
DOFULL = 1;
for t=1:40^DOFULL
    best_err = Inf;
    %Find feature that gives lowest classification error
    for xmin=0:1/6:(5/6)^DOFULL
        for ymin=0:(1/6):(5/6)^DOFULL
            for wid=1/6:1/6:(1-xmin)^DOFULL
                for hgt=1/6:1/6:(1-ymin)^DOFULL
                    FR = [xmin,ymin,wid,hgt];
                    for vert=0:DOFULL
                        for order=1:4^DOFULL
                            %Compute Feature
                            F = rectfeature(training_imdata,[face_rects(1:126,:),non_face_rects(1:126,:)],FR,order,vert);
                            %Calculate error
                            [theta,pola,err] = bestthreshold(F,Y,W);
                            
                            %Check if we found a new best error yet
                            if err < best_err
                                best_err = err;
                                classifiers(t,:) = [xmin,ymin,wid,hgt,vert,order,theta,pola];
                            end
    end;end;end;end;end;end;

    %adjust weights
    beta(t) = best_err / (1 - best_err);
    xmin = classifiers(t,1);
    ymin = classifiers(t,2);
    wid = classifiers(t,3);
    hgt = classifiers(t,4);
    vert = classifiers(t,5);
    order = classifiers(t,6);
    theta = classifiers(t,7);
    pola = classifiers(t,8);
    FR = [xmin,ymin,wid,hgt];
    F = rectfeature(training_imdata,[face_rects,non_face_rects],FR,order,vert);
    correct = (Y==(F .* pola < theta .* pola));
    for i = 1:126
        for j = 1:8
            if(correct(i,j))
                W(i,j) = W(i,j) .* beta(t);
            end
        end
    end
    W = W./sum(sum(W));
    
    %calculate adaboost weight
    alpha(t) = -log(beta(t));
    
    %print the results of this iteration
    disp(sprintf('t=%d (xmin,wid,ymin,hgt,vert,order,theta,pola,err)',t));
    disp(sprintf('(%f,%f,%f,%f,%f,%f,%f,%f,%f)\n',xmin,wid,ymin,hgt,vert,order,theta,pola,best_err));
    %disp(sprintf('(%f)\n',best_unweighted));
end          

save('learned_classifiers.txt','classifiers','-ascii')
%% Testing
%Binary Class Labels
Y = [ones(42,4) zeros(42,4)];

decision_sum = zeros(42, 8);

strong_errors = zeros(40,1);
weak_errors = zeros(40,1);

%strong classifiers
for t = 1:40^DOFULL
    xmin = classifiers(t,1);
    ymin = classifiers(t,2);
    wid = classifiers(t,3);
    hgt = classifiers(t,4);
    vert = classifiers(t,5);
    order = classifiers(t,6);
    theta = classifiers(t,7);
    pola = classifiers(t,8);
    
    FR = [xmin,ymin,wid,hgt];
    F = rectfeature(testing_imdata,[face_rects(127:168,:),non_face_rects(127:168,:)],FR,order,vert);
    
    decision = zeros(42,8);
    
    for i = 1:42
        for j = 1:8
            if(F(i,j) * pola < theta * pola)
                decision(i,j) = 1;
            else
                decision(i,j) = -1;
            end
        end
    end
    decision_sum = decision_sum + decision .* alpha(t);
    
    correct = ((decision_sum >= 0) .* (Y==1)) + ((decision_sum < 0) .* (Y==0));
    accuracy = sum(sum(correct)) ./ (42 .* 8);

    disp(sprintf('The unweighted error rate of the %dth strong classifier is %f\n',t,1-accuracy));
    strong_errors(t) = 1-accuracy;

end

%weak classifiers
for t = 1:40^DOFULL
    xmin = classifiers(t,1);
    ymin = classifiers(t,2);
    wid = classifiers(t,3);
    hgt = classifiers(t,4);
    vert = classifiers(t,5);
    order = classifiers(t,6);
    theta = classifiers(t,7);
    pola = classifiers(t,8);
    
    FR = [xmin,ymin,wid,hgt];
    F = rectfeature(testing_imdata,[face_rects(127:168,:),non_face_rects(127:168,:)],FR,order,vert);
    
    decision = zeros(42,8);
    
    for i = 1:42
        for j = 1:8
            if(F(i,j) * pola < theta * pola)
                decision(i,j) = 1;
            else
                decision(i,j) = -1;
            end
        end
    end
    
    correct = ((decision >= 0) .* (Y==1)) + ((decision < 0) .* (Y==0));
    accuracy = sum(sum(correct)) ./ (42 .* 8);

    disp(sprintf('The unweighted error rate of the %dth weak classifier is %f\n',t,1-accuracy));
    weak_errors(t) = 1-accuracy;

end

%weighted weak classifier error rates
for t = 1:40^DOFULL
    disp(sprintf('The weighted error of the %dth weak classifier is %f\n',t,beta(t)/(1+beta(t))));
end

figure;
plot(1:40,strong_errors);
title('Unweighted Strong Classifier Error Rate');
xlabel('Iteration');
ylabel('Error Rate');
figure;
plot(1:40,weak_errors);
title('Unweighted Weak Classifier Error Rate');
xlabel('Iteration');
ylabel('Error Rate');
figure;
plot(1:40,beta(t)/(1+beta(t)));
title('Weighted Weak Classifier Error Rate');
xlabel('Iteration');
ylabel('Error Rate');

%% EC 
II = cat(3, training_imdata, testing_imdata);
actual = cat(3, training_actual_im, testing_actual_im);
ec(classifiers, II, actual, face_rects, non_face_rects, DOFULL);


end
