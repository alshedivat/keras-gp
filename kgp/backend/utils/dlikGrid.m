function dlik = dlikGrid(hyp, mean, cov, lik, dcov, x, y, opt)
% dlikGrid   Derivative of the log marginal likelihood w.r.t. inputs.
%            Uses the MSGP approximations for computing kernel inverses.

% -----------------------------------------------------------------------------
% Do all the necessary parameter extraction as infGrid does.
% -----------------------------------------------------------------------------
if ~isfield(hyp,'mean'), hyp.mean = []; end % set an empty mean hyper parameter

if nargin<8, opt = []; end                          % make sure parameter exists
xg = cov{3}; p = numel(xg);                 % extract underlying grid parameters
[ng,Dg] = apxGrid('size',xg); N = prod(ng); D = sum(Dg);        % dimensionality
if isfield(opt,'proj'), proj = opt.proj; else proj = 'none'; end    % projection
hP = 1;                                                   % no projection at all
if isfield(hyp,'proj')                 % apply transformation matrix if provided
  hP = hyp.proj;
  if strncmpi(proj,'orth',4)
    hP = sqrtm(hP*hP'+eps*eye(D))\hP;                    % orthonormal projector
  elseif strncmpi(proj,'norm',4)
    hP = diag(1./sqrt(diag(hP*hP')+eps))*hP;                  % normal projector
  end
end
if isfield(opt,'deg'), deg = opt.deg; else deg = 3; end       % interpol. degree
% no of samples for covariance hyperparameter sampling approximation,
% see Po-Ru Loh et.al.: "Contrasting regional architectures of schizophrenia and
% other complex diseases using fast variance components analysis, biorxiv.org
if isfield(opt,'ndcovs'), ndcovs = max(opt.ndcovs,20);
else ndcovs = 0; end
[K,~] = feval(cov{:}, hyp.cov, x*hP');    % evaluate covariance mat constituents
m = feval(mean{:}, hyp.mean, x*hP');                      % evaluate mean vector
% TODO: Fix the problem with infLaplace.
if iscell(lik), lstr = lik{1}; else lstr = lik; end
if isa(lstr,'function_handle'), lstr = func2str(lstr); end
if isequal(lstr,'likGauss'), inf = @infGaussLik; else inf = @infLaplace; end
% inf = @infGaussLik;
[post nlZ] = inf(hyp, mean, cov, lik, x*hP', y, opt);
% -----------------------------------------------------------------------------

% Construct the derivative of interest
dlik = deriv_x(post.alpha, hP, K, xg, mean, hyp, x, deg);
end
