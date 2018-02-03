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

