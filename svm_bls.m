function [w, b, iter, w_list, b_list, acc] = svm_bls(X, y, X_test, y_test, lambda, num_iters, alpha, beta, t0)
% Achieve hinge loss + back line search
% X: data matrix, with each row being a sample
% y: label vector with elements -1 or 1
% lambda: 正则化参数
% num_iters: 最大迭代次数
% alpha: Backtracking Line Search 参数，0 < alpha < 1/2
% beta: Backtracking Line Search 参数，0 < beta < 1
% t0: Backtracking Line Search 截止条件

[m, n] = size(X);
w = zeros(n, 1); % initial weight
b = 0;           % initial bias
w_list = [];     % save the weights after each complete iteration
b_list = [];     % save the biases after each complete iteration
acc = zeros(num_iters, 1); % save the accuracies after each complete iteration

for iter = 1:num_iters
    prev_w = w; % save last iteration w
    prev_b = b; % save last iteration b

    % go through all the samples
    for i = 1:m
        xi = X(i, :)';
        yi = y(i);

        % calculate the loss
        loss = lambda * 0.5 * (w' * w) + max(0, 1 - yi * (w' * xi + b));
        
        % check if meet the constrain
        condition = yi * (w' * xi + b) < 1;
        
        % calculate derivative
        if condition
            w_grad = lambda * w - yi * xi;
            b_grad = -yi;
        else
            w_grad = lambda * w;
            b_grad = 0;
        end

        % Backtracking Line Search
        t = 1;
        while true
            % update parameter
            w_new = w - t * w_grad;
            b_new = b - t * b_grad;
            
            % calculate new loss
            new_loss = lambda * 0.5 * (w_new' * w_new) + max(0, 1 - yi * (w_new' * xi + b_new));
            
            % check
            if new_loss <= loss - alpha * t * (norm(w_grad)^2 + b_grad^2)
                break;
            end
            
            % reduce step
            t = beta * t;
            if t < t0
                % if the step size is too small, 
                % fill the remaining acc as the most recent valid value
                y_pre = svm_predict(w, b, X_test);
                acc_0 = svm_report(y_test, y_pre);
                acc(iter:end) = acc_0;
                return;
            end
        end

        % update parameter
        w = w_new;
        b = b_new;
    end

    % save the latest w and b after each complete traverse
    w_list = [w_list, w];
    b_list = [b_list, b];

    % calculate test set accuracy using the latest w and b
    y_pre = svm_predict(w, b, X_test);
    acc(iter) = svm_report(y_test, y_pre);
end

end