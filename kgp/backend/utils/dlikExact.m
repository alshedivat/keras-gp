function dlik = dlikExact(hyp, mean, cov, lik, dcov, x, y, opt)
% dlikExact_dcov   Derivative of the log marginal likelihood w.r.t. inputs.

if ~isfield(hyp,'mean'), hyp.mean = []; end       % check the hyp specification
if isempty(mean), mean = {@meanZero}; end                    % set default mean
if ischar(mean) || isa(mean, 'function_handle'), mean = {mean}; end % make cell
if ischar(cov) || isa(cov,'function_handle'), cov  = {cov};  end    % make cell

[n, D] = size(x);
sn2 = exp(2*hyp.lik);                              % noise variance of likGauss
if sn2<1e-6, sl = 1; else sl = sn2; end

% Get the necessary components for dlik from the inference procedure
post = infExact(hyp, mean, cov, 'likGauss', x, y);
alpha = post.alpha;
L = post.L;

% Compute dlik_dcov
dlik_dcov = solve_chol(L,eye(n))/sl - alpha*alpha';

% Compute dcov_dx and dcov_dz (note that dcov should be defined)
dcov_dx = dcov(hyp, x, []);

% Compute dlik as a proper contraction of dlik_dcov with dcov_dx
dlik = reshape(sum(bsxfun(@times, dlik_dcov, dcov_dx), 2), [n D]);
end
