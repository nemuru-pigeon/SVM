function [w, b, acc] = svm_gd(X, y, X_test, y_test, lambda, t, num_iters)
    % This function trains a linear SVM using stochastic gradient descent (SGD).
    % Inputs:
    %   X: Input data matrix, where each row represents a sample.
    %   y: Label vector, with elements being -1 or 1.
    %   X_test: Test data matrix for evaluation.
    %   y_test: Test label vector for evaluation.
    %   lambda: Regularization parameter.
    %   t: Initial learning rate (step size).
    %   num_iters: Number of iterations for training.
    %
    % Outputs:
    %   w: Optimized weight vector.
    %   b: Optimized bias term.
    %   acc: Accuracy recorded after each iteration on the test set.

    [m, n] = size(X); % m: number of samples, n: number of features.
    w = zeros(n, 1);  % Initialize weight vector.
    b = 0;            % Initialize bias term.
    acc = zeros(num_iters, 1); % Initialize accuracy tracking.

    for iter = 1:num_iters
        % Optional: Dynamically adjust the learning rate with linear decay.
        current_t = t / (1 + iter / num_iters);
        % Fixed learning rate alternative:
        % current_t = t;

        % Loop through all samples to update w and b.
        for i = 1:m
            xi = X(i, :)'; % Extract feature vector of the current sample.
            yi = y(i);     % Extract label of the current sample.

            % Check if the hinge loss condition yi * (w' * xi + b) >= 1 is satisfied.
            if yi * (w' * xi + b) < 1
                % Update w and b considering the misclassified case.
                w = w - current_t * (lambda * w - yi * xi);
                b = b - current_t * (-yi);
            else
                % Update w only with the regularization term.
                w = w - current_t * lambda * w;
                % b remains unchanged.
            end
        end

        % Compute and store the current accuracy on the test set.
        y_pred = svm_predict(w, b, X_test); % Custom prediction function.
        acc(iter) = svm_report(y_test, y_pred); % Custom evaluation function.
    end
end
