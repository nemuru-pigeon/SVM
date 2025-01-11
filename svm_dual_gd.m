function [w, b, acc] = svm_dual_gd(X, y, X_test, y_test, eta, num_iters)
    % Implementation of the projected gradient method for the dual problem of hard-margin SVM
    % X: Input data matrix, where each row is a sample
    % y: Label vector (-1 or 1)
    % X_test: Test set data
    % y_test: Test set labels
    % eta: Initial learning rate
    % num_iters: Maximum number of iterations

    [m, n] = size(X);        % Number of samples (m) and features (n)
    lambda = zeros(m, 1);     % Initialize Lagrange multipliers
    acc = zeros(num_iters, 1); % Array to store accuracy on the test set

    for iter = 1:num_iters
        % Decaying learning rate
        eta_t = eta / (1 + iter / 10);

        % Update each lambda_i
        for i = 1:m
            % Compute the gradient
            grad = 1 - y(i) * sum(lambda .* y .* (X * X(i, :)'));

            % Gradient update
            lambda(i) = lambda(i) + eta_t * grad;

            % Project onto [0, +∞)
            lambda(i) = max(0, lambda(i));
        end

        % Adjustment to satisfy the constraint: sum(lambda_i * y_i) = 0
        delta = sum(lambda .* y);
        lambda = lambda - y * (delta / sum(y.^2));

        % Update w
        w = sum((lambda .* y) .* X, 1)'; % w = sum(lambda_i * y_i * x_i)

        % Update bias term b
        support_indices = find(lambda > 1e-5); % Identify support vectors
        if ~isempty(support_indices)
            b = mean(y(support_indices) - X(support_indices, :) * w);
        else
            b = 0; % This should not occur in theory
        end

        % Predict on the test set
        y_pre = svm_predict(w, b, X_test);

        % Record test set accuracy
        acc(iter) = svm_report(y_test, y_pre);
    end
end
