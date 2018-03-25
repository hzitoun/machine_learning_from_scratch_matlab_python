function p = originalGaussian(X, mu, Sigma2)
%
%p(x)=p(x1;mu1,sig²1) * p(x2;mu2,sig²2) ... p(xn;mun,sig²n)
%
p =  (1./(sqrt(2 .* pi .* Sigma2))) .* exp(- ((X - mu) .^ 2) ./ (2 * Sigma2));
p = prod(p, 2); %  products of each row
end