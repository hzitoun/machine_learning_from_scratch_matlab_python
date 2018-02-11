%Always use vectorization instead of a for loop
%to compute gradient descent for linear regression
%for j
% wj = wj - alpha * 1/m * SUMi (h(xi) - yi) * xji
%instead do
% W = W - alpha * 1/M * SUMi (h(xi) - yi) * Xi

A = magic(3)
A(1:3, 1:2)