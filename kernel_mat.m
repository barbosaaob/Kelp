function K = kernel_mat(x, options)
%% Computes kernel matrix
%
% K = kernel_mat(x, options);
%
% Input:
%   x       : data matrix, n (instances) by d (dimension)
%   options : kernel options
%     options.param       : kernel parameter (\sigma^2)
%     options.kernel_type : kernel type (gaussian or dotproduct)
%
% Returns:
%   K       : kernel matrix

if ~isfield(options, 'kernel_type')
    options.kernel_type = 'gaussian';
end
if ~isfield(options, 'param')
    options.param = 2 * (sum(var(x)) / size(x, 2));
end

switch (options.kernel_type)
case 'gaussian'
        K = gaussian_kernel(x, options.param);

    case 'dotproduct'
        warning('*** Kernel not implemented. Computing gaussian kernel.');
        K = gaussian_kernel(x, options.param);

    otherwise
        warning('*** Invalid kernel type. Computing gaussian kernel.');
        K = gaussian_kernel(x, options.param);
end
end

function K = gaussian_kernel(x, param)
%% Computes gaussian kernel matrix
%

sqdist = pdist2(x, x).^2;
K = exp(-sqdist / (2 * param));
end
