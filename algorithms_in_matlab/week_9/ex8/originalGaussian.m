function p = originalGaussian(X, mu, Sigma2)
%
%p(x)=p(x1;?1,?21) * p(x2;?2,?22)?p(xn;?n,?2n)
%
n = size(X, 2);
p = 1;
for i=1:n
  p = p .* (1/(sqrt(2 .* pi .* Sigma2(i)))) .* exp(- ((X(:, i) - mu(i)) .^ 2) / (2 * Sigma2(i)));
end
end