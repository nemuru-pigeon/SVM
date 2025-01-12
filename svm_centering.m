function [w, b, iter] = svm_centering(X, y, w, b, t, num_iters, step_size, tol)
% X: input data, features
% y: label, 1 or -1
% w: weight
% b: bias
% t: the parameter of barrier method
% num_iters: max number of iterations
% step_size: step size
% tol: 

[m, n] = size(X);
% step_size = 0.01;

% parameters for back line search
alpha = 0.25;
beta = 0.5;
step_size0 = 1e-5;

old_w = zeros(n, 1);
old_b = 0;

for iter = 1:num_iters
    w_grad = t * w;
    b_grad = 0;
    loss = t * 0.5 * (w' * w);

    % go through all data points
    for i = 1:m
        xi = X(i, :)';
        yi = y(i);

        % calculate loss
        loss = loss - log(yi * (w' * xi + b) - 1);
        
        % calculate derivative
        w_grad = w_grad - yi * xi / (yi * (w' * xi + b) - 1);
        b_grad = b_grad + yi / (1 - yi * (w' * xi + b));
    end

    % backtracking line search
    while true
        % renew parameters
        w_new = w - step_size * w_grad;
        b_new = b - step_size * b_grad;
        
        % calculate new loss
        new_loss = t * 0.5 * (w_new' * w_new);
        for i = 1:m
            xi = X(i, :)';
            yi = y(i);
    
            new_loss = new_loss - log(yi * (w_new' * xi + b_new) - 1);
        end
        
        % check
        if new_loss < loss - alpha * step_size * (w_grad' * w_grad + b_grad^2)
            break;
        end
        
        % reduce step size
        step_size = beta * step_size;
        if step_size < step_size0
            return;
        end
    end

    % renew
    w = w_new;
    b = b_new;
    
    % % 更新参数
    % w = w - step_size * w_grad;
    % b = b - step_size * b_grad;

    if norm(old_w - w) < tol && abs(old_b - b) < tol
        return;
    end
    old_w = w;
    old_b = b;
end

end
