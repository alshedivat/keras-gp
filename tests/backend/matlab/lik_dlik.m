function [l dl] = lik_dlik(x, y, hyp, inf, mean, cov, dcov, lik, dlik)
  l = gp(hyp, inf, mean, cov, lik, x, y);
  if nargout > 1
    dl = feval(dlik{:}, hyp, mean, cov, lik, dcov, x, y);
  end
end
