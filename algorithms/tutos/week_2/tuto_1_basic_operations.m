%this is a comment
my_var = pi; %semicolon supressing output
disp(sprintf('2 decimals: %0.2f', my_var));
format short
my_var
format long
my_var

%Matrices
A = [1 2;3 4; 5 6] % (3,2) matrix
A = [1 2;
    3 4;
    5 6]
%Vectors
V = [1; 2; 3] % 3 dimensional vector
V = 1:6 % generate (1, m) vector
V = 0:2:8 % generate (1, m) vector min:step:max

M = ones(2,3) % generates  (2,3) 1-everywhere matrix

W = zeros(1,3) % generates (1,3) 0-everywhere matrix

W = rand(1,3) % generates (1,3) uniformal-rand(from 0 to 1)-everywhere matrix

W = randn(1,3) % generates (1,3)normal-rand(from 0 to 1)-everywhere matrix



W = -6 + sqrt(10) * (randn(1, 10000));
hist(W, 50);

I = eye(4) % 4 x 4 identity matrix

%help command
help eye



