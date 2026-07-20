function [cv_true, cv_perm, cv_rand, lasso_idx] = LIANA_OPLSDA_CV(X,Y,fullVarNames,kfold,ncomp,niter)
%% cross-validation framework for Datta et al (2026)
rng(1); % set random number generator seed
% niter is number of times to run cross validation
% kfold cross validation
% ncomp is number of model components

% initialize Y prediction matrices
Y_predicted = zeros(size(Y));
Yperm_predicted = zeros(length(Y),width(Y),10);
Yrand_predicted = zeros(length(Y),width(Y),10);

Yperm = zeros(size(Yperm_predicted));

% initialize CV score vectors
cv_true = zeros(niter,1);
cv_perm = zeros(niter,10);
cv_rand = zeros(niter,10);

lasso_idx = zeros(niter,width(X));

ia = 0; % initialize ia to an arbitrary number

% Loop through several independent CV iterations (defined by niter)
for i = 1:niter
    
    % partition the data
    cvp = cvpartition(height(X),'KFold',kfold);
    disp(append('Starting iteration #',string(i),'...'))
    tic

    % train models separately for each fold in cvp (defined by kfold)
    for j = 1:kfold
        disp(append('Starting fold #',string(j),'...'))

        % partition training and testing data sets
        [training_set] = training(cvp,j);
        [testing_set] = test(cvp,j);
        
        % independent feature selection for each fold of training data
        [~,ia] = run_elastic_net(X(training_set,:), Y(training_set,:), fullVarNames, 'minMSE',1, 100, 0.9, kfold);

        % print the lasso selected indices
        ia

        % if the feature selection fails, exit this loop and move on to the
        % next CV fold
        if length(ia)<=1
            disp('Invalid CV partition! Trying again...')
            Y_predicted(testing_set,:) = nan(size(Y(testing_set,:)));
            Yperm_predicted(testing_set,:,:) = nan(length(Y(testing_set,:)),width(Y(testing_set,:)),10);
            Yrand_predicted(testing_set,:,:) = nan(length(Y(testing_set,:)),width(Y(testing_set,:)),10);
            continue
        else 

            lasso_idx(i,ia) = 1; % flag features that got selected by LASSO
            
            % train  model using true LASSO features
            [~,~,~,~,BETA_true,~,~,~] = plsregress(X(training_set,ia),Y(training_set,:),ncomp,'cv','resubstitution');
            Y_predicted(testing_set,:) = [ones(size(X(testing_set,ia),1),1) X(testing_set,ia)]*BETA_true;

            % train model on equal number of random features
            for k = 1:10
                ib = randsample(width(X),length(ia));
                [~,~,~,~,BETA_rand,~,~,~] = plsregress(X(training_set,ib),Y(training_set,:),ncomp,'cv','resubstitution');
                Yrand_predicted(testing_set,:,k) = [ones(size(X(testing_set,ib),1),1) X(testing_set,ib)]*BETA_rand;
            end
            
            % train null model(s) with shuffled labels
            for k = 1:10
                Yperm(:,:,k) = Y(randperm(height(Y)),:);
                % [~,ia] = run_elastic_net(X(training_set,:), Yperm(training_set,:,k), fullVarNames, 'minMSE',1, 100, 0.9, kfold);
                % if length(ia)<=1
                %     disp('Yperm failed')
                %     Yperm_predicted(testing_set,:,k) = nan(length(Y(testing_set,:)),width(Y(testing_set,:)),1);
                %     continue
                % else
                [~,~,~,~,BETA_perm,~,~,~] = plsregress(X(training_set,ia),Yperm(training_set,:,k),ncomp,'cv','resubstitution');
                Yperm_predicted(testing_set,:,k) = [ones(size(X(testing_set,ia),1),1) X(testing_set,ia)]*BETA_perm;
                % end
            end
        
        end
        
    end
    
    toc
    
    % calculate CV accuracy for true model after all CV folds are complete
    cv_true(i) = cv_accuracy(Y,Y_predicted);

    % calculate CV accuracy for null models for each indepedent
    % trial (of 10) shuffling labels or choosing random features
    for k = 1:10
        cv_perm(i,k) = cv_accuracy(Yperm(:,:,k),Yperm_predicted(:,:,k));
        cv_rand(i,k) = cv_accuracy(Y,Yrand_predicted(:,:,k));
    end

end

% average across the repetitions of label permutation and random features
cv_perm = mean(cv_perm,2);
cv_rand = mean(cv_rand,2);

end