%% Breast Cancer LASSO Exploration
%% Prepare workspace

close all
clear

load BreastCancer

%%  10-fold CV 

% each row of setindices denotes the starting an ending index for one
% partition of the data: 5 sets of 30 samples and 5 sets of 29 samples
setindices = [1,30;31,60;61,90;91,120;121,150;151,179;180,208;209,237;238,266;267,295];

% each row of holdoutindices denotes the partitions that are held out from
% the training set
holdoutindices = [1,2;2,3;3,4;4,5;5,6;7,8;9,10;10,1];

cases = size(holdoutindices,1);

% be sure to initiate the quantities you want to measure before looping
% through the various training, validation, and test partitions
%
lam_vals = [1e-6 1e-4 1e-2 1e-1 logspace(0,2,20)];
squareerror_lasso = zeros(1,cases);
error_lasso = zeros(1,cases);
squareerror_rr = zeros(1,cases);
error_rr = zeros(1,cases);

% Loop over various cases
for j = 1:cases
    disp('Cases: ')
    j
    % row indices of first validation set
    v1_ind = setindices(holdoutindices(j,1),1):setindices(holdoutindices(j,1),2);
    
    % row indices of second validation set
    v2_ind = setindices(holdoutindices(j,2),1):setindices(holdoutindices(j,2),2);
    
    % row indices of training set
    trn_ind = setdiff(1:295,[v1_ind, v2_ind]);
    
    % define matrix of features and labels corresponding to first
    % validation set
    Av1 = X(v1_ind,:);
    bv1 = y(v1_ind);
    
    % define matrix of features and labels corresponding to second
    % validation set
    Av2 = X(v2_ind,:);
    bv2 = y(v2_ind);
    
    % define matrix of features and labels corresponding to the 
    % training set
    At = X(trn_ind,:);
    bt = y(trn_ind);

% Use training data to learn classifier
    W = ista_solve_hot(At,bt,lam_vals);
    [m_w,n_w] = size(W);
    soln = zeros(1,n_w);
    for i=1:n_w
        soln(1,i) = norm(sign(Av1*W(:,i))-bv1)^2 + lam_vals(i)*norm(W(:,i),1);
    end
    
    [val,ind] = min(soln);
    final_result = sign(Av2*W(:,ind));
    squareerror_lasso(j) = norm(final_result-bv2)^2;
    error_lasso(j) = sum(final_result~=bv2);
    
    [f,g] = size(bv2);
    display('with LASSO: ')
    display('prediction error with each w is:')
    soln
    display('squared error on final subset with best lambda is: ')
    squareerror_lasso(j)
    display('test error on final subset with best lambda is: ')
    error_lasso(j)
    display('error rate: ')
    error_lasso(j)/f
    
    % ridge regression
    soln = zeros(1,n_w);
    M = transpose(At)*bt;
    N = transpose(At)*At;
    
    for k=1:n_w
        lambda = lam_vals(k);
        W = (N+lambda*eye(m_w))\M;
        soln(1,k) = norm(sign(Av1*W)-bv1)^2+lambda*norm(W)^2;
        weights(:,k)=W;
    end
    [val,ind]=min(soln);
    final_result = sign(Av2*weights(:,ind));
    squareerror_rr(j) = norm(final_result-bv2)^2;
    error_rr(j)=sum(final_result~=bv2);
    
    display('with ridge regression: ')
    display('prediction error with each w is:')
    soln
    display('squared error on final subset with best lambda is: ')
    squareerror_rr(j)
    display('test error on final subset with best lambda is: ')
    error_rr(j)
    display('error rate: ')
    error_rr(j)/f
end

mean_squareerror_rr = mean(squareerror_rr)
mean_error_rr = mean(error_rr)
mean_squareerror_lasso = mean(squareerror_lasso)
mean_error_lasso = mean(error_lasso)



    
    
    
    
