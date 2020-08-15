%% Assignment 6
%% Prepare workspace

close all
clear
X = csvread('sdata.csv');

%% Display data

% Use rotate tool in the figure to view data from different angles
figure
scatter3( X(:,1), X(:,2), X(:,3), 'r.', 'LineWidth', 3 )
xlabel('x_1')
ylabel('x_2')
zlabel('x_3')

hold on
scatter3(0,0,0,'MarkerEdgeColor','k',...
        'MarkerFaceColor','k')
hold off
title('Data points (red), origin (black)')
view(70,30)


%% Remove mean

mn = mean(X);

Xz = X - ones(1000,1)*mn;

figure
scatter3( Xz(:,1), Xz(:,2), Xz(:,3), 'r.', 'LineWidth', 3 )
xlabel('x_1')
ylabel('x_2')
zlabel('x_3')

hold on
scatter3(0,0,0,'MarkerEdgeColor','k',...
        'MarkerFaceColor','k')
hold off
title('Mean removed data points (red), origin (black)')
view(70,30)

%% Take SVD to find best line

[U,S,V] = svd(Xz,'econ');

 a = V(:,1);  % Complete this line

%% Display best line on scatterplot

figure
scatter3( Xz(:,1), Xz(:,2), Xz(:,3), 'r.', 'LineWidth', 3 )
xlabel('x_1')
ylabel('x_2')
zlabel('x_3')

a2 = V(:,2);
title('Mean removed data points (red), 1D Subspace Approx (blue)')

% Scale length of line by root-mean-square of data for display
scale1 = S(1,1)/sqrt(size(Xz,1));
scale2 = S(2,2)/sqrt(size(Xz,1));

hold on

plot3(scale1*[0;a(1)],scale1*[0;a(2)],scale1*[0;a(3)], 'b', 'LineWidth', 4)
plot3(scale2*[0;a2(1)],scale2*[0;a2(2)],scale2*[0;a2(3)], 'b', 'LineWidth', 4)
hold off

view(70,30)

Xz_2 = U(:,1)*S(1,1)*transpose(a) + U(:,2)*S(2,2)*transpose(a2);
figure
scatter3( Xz(:,1), Xz(:,2), Xz(:,3), 'r.', 'LineWidth', 3 )
hold on
scatter3( Xz_2(:,1), Xz_2(:,2), Xz_2(:,3), 'b.', 'LineWidth', 3 )
xlabel('x_1')
ylabel('x_2')
zlabel('x_3')
hold off
%% Problem 3 part a
load('face_emotion_data.mat')
numRows = 0;
error = zeros(8,7);
for i=1:8
    heldoutx = X(numRows+1:numRows+16,:);
    heldouty = y(numRows+1:numRows+16,:);
    trainingx = X([1:numRows numRows+17:128],:);
    trainingy = y([1:numRows numRows+17:128],:);
    numRows = numRows+16;
    numRows_2 = 0;
    
    for j=1:7
        finaltrainingx = trainingx([1:numRows_2 numRows_2+17:112],:);
        finaltrainingy = trainingy([1:numRows_2 numRows_2+17:112],:);
        testx = trainingx(numRows_2+1:numRows_2+16,:);
        testy = trainingy(numRows_2+1:numRows_2+16,:);
        numRows_2 = numRows_2+16;
        
        [U,S,V] = svd(finaltrainingx);
        S_inverse = zeros(size(S));
        
        error_1 = zeros(1,9);
        error_2 = zeros(1,9);
        
        for k=1:9
            for r=1:k
                S_inverse(r,r) = 1/S(r,r);
            end
            w = V*transpose(S_inverse)*transpose(U)*finaltrainingy;
            y_predict_1 = sign(testx*w);
            error_1(1,k)=sum(y_predict_1 ~= testy)/16;
            y_predict_2 = sign(heldoutx*w);
            error_2(1,k)=sum(y_predict_2 ~= heldouty)/16;
        end
        [e_min,k_min] = min(error_1);
        error(i,j) = error_2(1,k_min);
    end
end
soln = mean(mean(error));
display(soln);
%% partb 3
lambda=[0;0.5;1;2;4;8;16];
error = zeros(8,7);

numRows=0;
for i=1:8
    heldoutx = X(numRows+1:numRows+16,:);
    heldouty = y(numRows+1:numRows+16,:);
    trainingx = X([1:numRows numRows+17:128],:);
    trainingy = y([1:numRows numRows+17:128],:);
    numRows = numRows+16;
    numRows_2 = 0;
    
    for j=1:7
        finaltrainingx = trainingx([1:numRows_2 numRows_2+17:112],:);
        finaltrainingy = trainingy([1:numRows_2 numRows_2+17:112],:);
        testx = trainingx(numRows_2+1:numRows_2+16,:);
        testy = trainingy(numRows_2+1:numRows_2+16,:);
        numRows_2 = numRows_2+16;
        
        [U,S,V] = svd(finaltrainingx);
        S_inverse = zeros(size(S));
        
        error_1 = zeros(1,9);
        error_2 = zeros(1,9);
        
        for k=1:7
            for r=1:9
                S_inverse(r,r) = S(r,r)/(S(r,r)^2+lambda(k));
            end
            w = V*transpose(S_inverse)*transpose(U)*finaltrainingy;
            y_predict_1 = sign(testx*w);
            error_1(1,k)=sum(y_predict_1 ~= testy)/16;
            y_predict_2 = sign(heldoutx*w);
            error_2(1,k)=sum(y_predict_2 ~= heldouty)/16;
        end
        [e_min,k_min] = min(error_1);
        error(i,j) = error_2(1,k_min);
    end
end
soln1 = mean(mean(error));
display(soln1);

            
            
        
        

