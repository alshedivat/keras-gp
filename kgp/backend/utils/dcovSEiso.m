function dcov_dx = dcovSEiso(hyp, x, z, opt)
% dcovSEiso  Compute the derivative of covSEiso kernel w.r.t. inputs.
%            Return a tensor of size [n, n, D].

% Compute the covariance kernel matrix
K = covSEiso(hyp.cov, x, z);
ell = exp(hyp.cov(1));                            % characteristic length scale

% Reshape vectors appropriately to feed into bsxfun
[n, D] = size(x);
if isempty(z), z = x; end
x = reshape(x, [n 1 D]);
z = reshape(z, [1 n D]);

% Compute the derivatives
dcov_dx = -bsxfun(@times, K / ell^2, bsxfun(@minus, x, z));

end
