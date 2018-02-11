V = zeros(10, 1)
%for
for i=1:10,
    V(i) = 2 ^ i;
end;
V
%while
i = 1;
while i <= 5,
    V(i) = 100;
    i = i+1;
end
V
%using break and continue
i = 1;
while true,
    V(i) = 999;
    i = i+ 1;
    if i == 6,
        break;
    end
end
V
%general syntax of if statement
V(1) = 2;
if V(1) == 1,
    disp('The value is one');
elseif V(1) ==2,
   disp('The value is two');
else
     disp('The value is not one or two');
end

%call local function
disp(fn(5));
%call function defined in a seperate file
myfuns = tuto_5_functions;
disp(myfuns.square_function(5));
%multi return
[a, b] = myfuns.squareAndCubeFunction(5);
disp(a)
disp(b)

%calling cost function
X = [1 1; 2 2; 1 3];
y = [1; 2; 3];
theta = [0; 1];

J = myfuns.costFunctionJ(X,y, theta)
theta = [0; 0];
J = myfuns.costFunctionJ(X,y, theta) 
% which is
J = (1^2 + 2^2 + 3^2) / (2*3)



%Add lib dir to octave search path
addpath('.');




%define function, in a script it has to be the last thing defined
function y = fn(x)
y = x ^ 2;
end
