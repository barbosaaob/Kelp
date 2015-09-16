function Y = kelp(X, Xs, Ys, options)
%% Computes Kelp projection
% http://www.lcad.icmc.usp.br/~barbosa
%
% y = kelp(x, xs, ys, options);
%
% d is the data dimension
% p is the visual space dimension
%
% Input:
%   X       : data matrix, n (instances) by d (dimensions)
%   Xs      : sample points data, k instances from X
%   Ys      : projection of sample points, k (instances) by p (dimensions) matrix
%   options : kernel options
%     options.param       : kernel parameter (sigma)
%     options.kernel_type : kernel type (gaussian or dotprod)
%
% Returns:
%   Y       : n (instances) by p (dimensions) matrix containing the data projection

[n, d] = size(X);   % n = number of instances, d = data dimension
[k, a] = size(Xs);  % k = sample size
p = size(Ys, 2);    % p = visual space dimension
assert(d == a, '*** Kelp ERROR: X and Xs dimension must be the same!')

% set default values if necessary
if ~isfield(options, 'kernel_type')
    options.kernel_type = 'gaussian';
end
if ~isfield(options, 'param')
    options.param = 2 * (sum(var(X)) / size(X, 2));
end

% compute kernel matrix
K_uncentered = kernel_mat(Xs, options);
K = K_uncentered;
% centering kernel matrix
In = ones(k)./k;
K = K - In*K - K*In + In*K*In;

% computes eigendecomposition of K
[alpha, lambda] = eig(K);
[lambda, perm] = sort(real(diag(lambda)), 'descend');  % sort eigenvalues
alpha = alpha(:, perm);                                % sort eigenvectors
% remove zero eigenvalues
nonzero_thold = 1e-6;
nonzero_idx = 0;
for i = 1:k
    if (lambda(i) < nonzero_thold)
        break;
    end
    nonzero_idx = i;
    alpha(:,i) = alpha(:,i) / sqrt(lambda(i));  % make eigenvectors have norm = 1
end

% computes inverse of Gamma matrix
lambda(1:nonzero_idx) = k ./ lambda(1:nonzero_idx);
lambda = diag(lambda);

% projection
Y  = zeros(n, p);
um = ones(k, 1) ./ k;
UM = ones(k) ./ k;
alpha = alpha(:,1:nonzero_idx);
lambda = lambda(1:nonzero_idx, 1:nonzero_idx);
P = (1 / k) * alpha * lambda * alpha' * K * Ys;
for pto = 1:n
    pnt = X(pto,:);
    aux = Xs - repmat(pnt, k, 1);
    Kx = exp(-dot(aux, aux, 2) / (2 * options.param));
    Kxc = Kx - K_uncentered * um - UM * Kx + UM * K_uncentered * um;
    Y(pto, :) = Kxc' * P;
end
