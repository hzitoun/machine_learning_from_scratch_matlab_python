A = [1 2; 3 4; 5 6];
B = [11 12; 13 14; 15 16];
C = [1 1; 2 2];

A * C 
A .* B %element wise op multi one by one

V  = [1; 2; 3]
1 ./ V
log(V)
exp(V)
abs(V)
-V % same as -1 * V
V + ones(length(V), 1) % add 1 to each element
%or directly
V + 1

A' %transpose (turn left)

max_val = max(V)
[max_val, max_index] = max(V)

%do elemnt wise comparaison, compare each elmt to 2
V < 2

%find all eltms inf to 3
find(V < 3)

M = magic(3)

[row, colum] = find(M >= 7)

%sum
sum(M)
%sum of each column
sum(M, 1)
%sum of each row
sum(M, 2)

%sum diag
sum(sum(M .* eye(3)))
disp('****')
sum(sum(M .* flipud(eye(3))))


%mult
prod(M)

W = [2.5 3.2 7.9]

%floor
floor(W)
ceil(W)

%max of each column
max(M, [], 1)

%max of each row
max(M, [], 2)

%max of matrix
max(M)

%max value
max(max(M)) % or max(M(:))
max(M(:))


%pseudo inverse
pinv(M)

identity = M * pinv(M)

