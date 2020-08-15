% Problem 1(e)
E = X*w; % for 9-features case
F = X_new*w_new; % for 3-features case

count = 0;
for i=1:length(y)
    if E(i)*y(i)<0
        count = count+1;
    end
end
disp(count/128);

count1 = 0;
for i=1:length(y)
    if F(i)*y(i)<0
        count1 = count1+1;
    end
end
disp(count1/128);

% Problem 1(f)
number_row = 0;
error_rates = 0;
X_new = X(:,[1 3:4]);

for i = 1:8
    testX = X_new(number_row+1:number_row+16,:);
    testy = y(number_row+1:number_row+16,:);
    trainingX = X_new([1:number_row number_row+17:128],:);
    trainingy = y([1:number_row number_row+17:128],:);
    w1 = ((transpose(trainingX)*trainingX)^-1)*transpose(trainingX)*trainingy;
    y_predict = testX*w1;
    number_row = number_row + 16;
    count = 0;
    for i = 1:16
        if y_predict(i)*y(i)<0
            count = count+1;
        end
    end
    misclassification = count/16;
    error_rates = error_rates + misclassification;
end

average_rate = error_rates/8;
disp(average_rate);

