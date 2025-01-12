function [w, b, alpha, w_list, b_list, acc] = svm_lag(X, y, X_test, y_test, eta, num_iters)
    % input:
    % X: input data
    % y: labels
    % eta: initial learning rate
    % num_iters: max number of iterations

    [m, n] = size(X); % number of points and number of features
    w = zeros(n, 1); % initial w
    b = 0; % initial b
    w_list = [];
    b_list = [];
    alpha = zeros(m, 1); % initial Lagrange multiplier
    acc = zeros(num_iters, 1); % initial accuracy

    for iter = 1:num_iters
        % dynamic adjustment of learning rate (optional)
        eta_t = eta / (1 + iter / 10); % learning rate attenuation
        grad_w = w - sum(alpha .* y .* X, 1)'; % derivative to w
        grad_b = -sum(alpha .* y); % derivative to b

        % go through
        for i = 1:m
            % dynamic computed gradient
            grad_alpha = 1 - y(i) * (X(i, :) * w + b); % 对 alpha_i 的梯度
            % update alpha_i and project to a non-negative region
            alpha(i) = max(0, alpha(i) + eta_t * grad_alpha);
        end

        w = w - eta_t * grad_w;
        b = b - eta_t * grad_b;

        % each iteration takes the last parameter
        w_list = [w_list, w];
        b_list = [b_list, b];
        
        % calculate the accuracy
        y_pre = svm_predict(w, b, X_test);
        acc(iter) = svm_report(y_test, y_pre);
    end
end
