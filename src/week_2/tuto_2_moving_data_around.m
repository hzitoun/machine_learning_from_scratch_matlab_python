A = [1 2;3 4; 5 6];
disp(A);
sz = size(A); % size is matrix!
disp(sz);
number_of_rows = size(A, 1);
disp(number_of_rows);
number_of_columns = size(A, 2);
disp(number_of_columns);

%vectors
V = [1 2 3 4];
size_longest_dimension = length(V); % 4
length([1;2;3;4;5]); % gives 5 (longest dim is number of rows = 5
%print working dir
pwd 
%change dir
cd .
%list files
ls
%load data
load featuresX.dat
load('featuresX.dat')
%see current variables
who
disp(featuresX)
size(featuresX)
%see current variables structured
whos

%get rid of variables
clear V

%save variabke to disk
save out.mat featuresX 
clear % clears everything
load out.mat %load variables to mem
whos

save textfile.txt featuresX -ascii %save as txt

%fetch data from matrix
A = [1 2; 3 4; 5 6];
A(3, 2)
A(2, :) % ":" means elt along that row/colum
A([1 3], :) % mean first and third row of A

A(:, 2)
A(:, 2) = [10; 11; 12] % replace the last column
A = [A, [100; 101; 102]] % append a new column vector to right

size(A) % new become 3 by 3

A(:) % put all elemt into a single vector

A = [1 2; 3 4; 5 6];
B = [11 12; 13 14; 15 16];

C = [A B] %concat the matrices horiz (same as [A, B]
C = [A; B] % concat verti
