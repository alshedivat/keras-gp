function dcov_dx = dcovSEard(hyp, x, z, opt)
% dcovSEard  Compute the derivative of covSEard kernel w.r.t. inputs.
%            Return a tensor of size [n, n, D].

% Compute the covariance kernel matrix
K = covSEard(hyp.cov, x, z);
ell = exp(hyp.cov(1));                            % characteristic length scale

% Reshape vectors appropriately to feed into bsxfun
[n, D] = size(x);
if isempty(z), z = x; end
x = reshape(x, [n 1 D]);
z = reshape(z, [1 n D]);

% Compute the derivatives
dcov_dx = -1/ell^2 * bsxfun(@times, K, bsxfun(@minus, x, z));

end
