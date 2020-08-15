%% Problem 1
clear
close all

edges = csvread('wisconsin_edges.csv');

node_count = max(edges(:))+1;

A = zeros(node_count,node_count);
[m,n] = size(edges);
for i=1:m
  from_node = edges(i,1);
  to_node = edges(i,2);
  A(to_node+1,from_node+1)=1;
end

A = A+0.001;
A = A./sum(A);
[vecs,vals]=eig(A);
vals = diag(vals);
[val,ind] = max(vals);
index = eigs(A,1);
vec = vecs(:,ind);
S = A*vec;
for loop=1:100
    S=A*S;
end
[aa,indices]=sort(S,'descend');
first_index = indices(1,1)
third_index = indices(3,1)
 
% b) The most important page is Dane County, Wisconsin
% c) The third most important page is Sauk County, Wisconsin



% Hint: use 
% eigs(A,k)
% where k=1 to get the first eigenvector, instead of 
% eig(A)
% as computation of all eigenvectors will take ~5 minutes

%% Problem 2
A = load('BreastCancer.mat');
x = A.X;
y = A.y;
vertical = [];
horizontal = [];
x1 = x(1:100,:);
y1 = y(1:100,1);
lambdas = [50,10,4.5,4,3.5,3,2,1,0.5,0];

weights = ista_solve_hot(x1,y1,lambdas);

for j=1:10
    lambda = lambdas(j);
    vertical(:,j) = norm(sign(x*weights(:,j))-y)^2;
    horizontal(:,j) = norm(weights(:,j),1);
end
plot(horizontal,vertical);
title('Problem 2a');

% According to the plot, the optimal weights go down when lambda increases.
% The plot shows that it becomes closer to zero in horizontal axis. The
% lowest point is the best solution. At the case when lambda is very small,
% weights will go up so that the error is larger.

%% Problem 2b
weights1 = weights;
weights1(abs(weights1) <= 10^(-6)) = 0;
sparsity = zeros(10,1);
error = zeros(10,1);

for j=1:10
    sparsity(j,1) = sum(weights1(:,j)==0);
    error(j,1) = sum(sign(x*weights(:,j))~=y)/295;
end
plot(sparsity,error);
title('Problem 2b');
% The error becomes larger when the sparsity increases. The number of
% features are smaller. When there is no regulation, the error rate is the
% highest. Also, sparsity is zero when there are a lot of regulations.

%% Problem 2c
vertical = [];
horizontal = [];
x1 = x(101:295,:);
y1 = y(101:295,1);
lambdas = [50,10,4.5,4,3.5,3,2,1,0.5,0];

weights = ista_solve_hot(x1,y1,lambdas);

for i=1:10
    lambda = lambdas(i);
    vertical(:,i) = norm(sign(x*weights(:,i))-y)^2;
    horizontal(:,i) = norm(weights(:,i),1);
end
figure;
plot(horizontal,vertical);
title('Problem 2c(1)');
% Optimal weights will go down when lambda increases. In particular, it's
% closer to zero in horizontal axis in the graph. The lowest point is the
% optimal. When lambda becomes small, optimal weights are almost the same
% (not change so much).

weights1 = weights;
weights1(abs(weights1) <= 10^(-6)) = 0;
sparsity = zeros(10,1);
error = zeros(10,1);

for j=1:10
    sparsity(j,1) = sum(weights1(:,j)==0);
    error(j,1) = sum(sign(x*weights(:,j))~=y)/295;
end
figure;
plot(sparsity,error);
title('Problem 2c(2)');
% When regulation exists, the error does not depend on sparsity or
% regulation very much. However, when there is lack of regulation, sparsity
% is large, and the error rate is large as well.
