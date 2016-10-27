function [c dc] = cov_dcov(x, hyp, cov, dcov, opt)
    [n, D] = size(x);
    c = feval(cov{:}, hyp.cov, x);
    if nargout > 1
        dcov_dx = dcov(hyp, x, [], opt);
        dc = zeros(n, n, n, D);
        for i=1:n
            dc(i,:,i,:) = dcov_dx(i,:,:);
            dc(:,i,i,:) = -dcov_dx(:,i,:);
        end
    end
end
