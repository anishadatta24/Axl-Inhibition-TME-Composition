%% This script performs multivariate analysis (LASSO-OPLSDA) for Datta et al, 2026 (CMBE).
%% Contributed by:  Remziye Wessel (Apr 2026)
%% Load LIANA data (CSV) 
%% You will need to install Remziye's PLSDA package from GitHub to run this code:
%% https://github.com/rewessel/PLSR-DA_MATLAB

% Or whatever your file path is...
cd 'Z:/users/wesselr/data/LIANA_OPLSDA_Anisha/'
data = readtable('11_LIANA_results_by_sample_for_OPLSDA.csv');

% Add PLSR-DA_MATLAB GitHub local directory to your path.
addpath('C:\Users\remzi\OneDrive\Documents\GitHub\PLSR-DA_MATLAB')

%% Data cleaning and preprocessing for LASSO-OPLSDA model

% Extract sample names
sample_names = unique(data.sample_new);

% Optionally, subset data to only injured/uninjured or co-culture
% conditions.
% sample_names = sample_names(contains(sample_names,'_Y_') & contains(sample_names,'co_')); % _Y_ = injured, _N_ = uninjured
% sample_names = sample_names(contains(sample_names,'_N_')); % _Y_ = injured, _N_ = uninjured

% This loop finds the interactions that are detected across all conditions.
% This is a necessary step to remove missing values present in raw output
% from LIANA (because LASSO-OPLSDA cannot handle missing values, unless you
% implement an imputation method.

for i = 1:length(sample_names)
    
    % This step subsets data to only the samples you wish to include.
    data_temp = data(strcmp(data.sample_new,sample_names(i)),:);

    % Create feature names from the L-R interactions and cell types in
    % LIANA data set.
    data_temp.interactions = append(data_temp.source,'-',data_temp.target,'-',data_temp.ligand_complex,'-',data_temp.receptor_complex);
    
    % X = data_temp(:,{'magnitude_rank','interactions'});
    % colnames = X.interactions;

    % Create a table of the interaction score data, using sample names to
    % label columns.
    T = array2table(data_temp.magnitude_rank);
    T.Properties.VariableNames = sample_names(i);
    T.interactions = data_temp.interactions;
   
    % If this is the first sample, no need to take a union of the data -
    % just save this table for use in the next iteration.
    if i == 1
        T_full = T;
    else
    % If this is not the first iteration, then join the table of
    % interactions present in this sample with the table of interactions
    % present across the experiment (this will grow on each iteration).
        T_full = innerjoin(T_full,T,"Keys",'interactions');
    end
end

% Name the rows, and normalize the interaction scores (1-magnitude rank).
T_full.Properties.RowNames = T_full.interactions;
T_full = 1-T_full(:,[1,3:width(T_full)]);

%% LASSO-OPLSDA - compare Bem vs DMSO ctrl samples.
%% This chunk of code runs LASSO and trains an OPLSDA model on the full data set.
%% There is a "cheating" cross validation step in this function, but a more formal 
%% cross-validation with independent feature selection on each training subset of the data
%% is in the subsequent code block.

% Extract variable names.
myVarNames = T_full.Properties.RowNames;

% Transpose, log-transform, and z-score X data set.
X = table2array(T_full)';
X = zscore(log10(X+1));

% Remove any features that are all NaN.
myVarNames(all(isnan(X), 1))=[];
X(:,all(isnan(X), 1))=[];

% Set up the outcome variable (Y).
conditions = T_full.Properties.VariableNames';
Y = ones(height(X),2);
Y(contains(conditions,'DMSO'),1)=0;
Y(contains(conditions,'Bem'),2)=0;

% Set your palette to color groups in the OPLSDA output plots.
myColors = [1 0 0;0 0 1];

% Run PLSDA. See rewessel/PLSR-DA_MATLAB for instructions on this code.
% Note - The example of how to call this function on the GitHub readme.md
% might be out of date...
mymodel = PLSDA_main(X,Y,2,myVarNames',{'yes',1,0.9},'orthogonal','',{'kfold',5},1000,{'Bem','DMSO'},myColors)

%% MORE FORMAL CROSS-VALIDATION
%% Cross validate the model, performing LASSO on each training data set separately
%% while holding out the test data set.

% See LIANA_OPLSDA_CV.m function for a description of these arguments.
[CV_results.cv_true, CV_results.cv_perm, CV_results.cv_rand, CV_results.lasso_idx] = ...
    LIANA_OPLSDA_CV(X,Y,myVarNames',5,2,100);

% Optionally save the results.
% save('CV_results.mat',"CV_results",'mymodel')

%% Plot the cross-validation results.
figure
clear violindata
violindata.TrueModel = CV_results.cv_true;
violindata.Rand = CV_results.cv_rand;
violindata.Perm = CV_results.cv_perm;

violindata.TrueModel(violindata.TrueModel==0) = [];
violindata.Rand(violindata.Rand==0) = [];
violindata.Perm(violindata.Perm==0) = [];

violinplot(violindata); hold on
yline(length(Y(Y(:,1)==1,1))/length(Y)*100,'linestyle','--')
ylabel('CV accuracy (%)'); xticks([1:3]); xticklabels({'True Model','Random Features','Permuted Y-labels'})

% Compare the true model vs. the model trained on random features and model
% with permuted labels.
[~,pval] = ttest2(violindata.TrueModel,violindata.Rand);
[~,pval] = ttest2(violindata.TrueModel,violindata.Perm);
