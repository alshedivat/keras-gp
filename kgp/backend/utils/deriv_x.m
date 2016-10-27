function dx = deriv_x(alpha, P, K, xg, mean, hyp, x, deg, h)
  % deriv_x   Compute dcovGrid_dx for given parameters.
  %           The function is taken from DKL framework.
  if ~exist('h', 'var') h = 1e-5; end
  if P == 1, tP = eye(size(x,2)); else tP = P; end
  xP = x*P';
  [M,dM] = covGrid('interp',xg,xP,deg);       % grid interp derivative matrices
  beta = K.mvm(M'*alpha);                         % dP(i,j) = -alpha'*dMij*beta
  dxP = [];
  for i=1:size(tP,2)
    if equi(xg,i)                                              % scaling factor
      wi = max(xg{i})-min(xg{i});
    else
      wi = 1;
    end
    xP(:,i) = xP(:,i) - h;
    dmi1 = feval(mean{:},hyp.mean,xP);
    xP(:,i) = xP(:,i) + 2*h;
    dmi2 = feval(mean{:},hyp.mean,xP);
    dmi_dx = (dmi2 - dmi1) / (2*h);   % numerical approximation to the gradient
    xP(:,i) = xP(:,i) - h;
    betai = dmi_dx + dM{i}*beta/wi;
    dxP = [dxP -alpha.*betai];
  end
  dx = dxP*tP;
end

function eq = equi(xg,i)                        % grid along dim i is equispaced
  ni = size(xg{i},1);
  if ni>1                              % diagnose if data is linearly increasing
    dev = abs(diff(xg{i})-ones(ni-1,1)*(xg{i}(2,:)-xg{i}(1,:)));
    eq = max(dev(:))<1e-9;
  else
    eq = true;
  end
end
