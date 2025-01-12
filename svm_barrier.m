function [w, b, total_iter, w_res, b_res] = svm_barrier(X, y, t, mu, tol, num_iters, step_size)
% X: data matrix, with each row being a sample
% y: label vector with elements -1 or 1
% t: the parameter for barrier method, larger is better
% mu: the multiplier for t
% tol: tolerance
% num_iters: max number of iterations
% step_size: step size

% initial parameters
[m, n] = size(X);
% w = zeros(n, 1);
% b = 0;
% using feasible initials
w = [5.17; -0.59];
b = -2.33;
total_iter = 0;
w_res = [];
b_res = [];

% start iteration
for iter = 1:num_iters
    % center step optimized using gradient descent
    [w, b, func_iter] = svm_centering(X, y, w, b, t, num_iters, step_size / t, 1e-5);
    disp(w);
    disp(b);
    plot_svm_decision_boundary(X, y, w, b);
    disp('----------------------------------');

    % record
    w_res = [w_res; w'];
    b_res = [b_res; b];
    total_iter = total_iter + func_iter;

    % check if meet the stop criterion
    if m / t < tol
        break;
    else
        t = mu * t;
    end
end

end
