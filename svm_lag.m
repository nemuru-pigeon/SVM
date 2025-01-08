function [w, b, alpha, w_list, b_list, acc] = svm_lag(X, y, X_test, y_test, eta, num_iters)
    % 输入：
    % X: 输入数据矩阵，每行是一个样本
    % y: 标签向量，取值为 -1 或 1
    % eta: 初始学习率
    % num_iters: 最大迭代次数

    [m, n] = size(X); % 样本数和特征数
    w = zeros(n, 1); % 初始化 w
    b = 0; % 初始化 b
    w_list = [];
    b_list = [];
    alpha = zeros(m, 1); % 初始化拉格朗日乘子
    acc = zeros(num_iters, 1); % 初始化误差记录

    for iter = 1:num_iters
        % 动态调整学习率（可选）
        eta_t = eta / (1 + iter / 10); % 学习率衰减
        grad_w = w - sum(alpha .* y .* X, 1)'; % 对 w 的梯度
        grad_b = -sum(alpha .* y); % 对 b 的梯度
        % 遍历每个样本
        for i = 1:m
            % 动态计算梯度
            grad_alpha = 1 - y(i) * (X(i, :) * w + b); % 对 alpha_i 的梯度
            % 更新 alpha_i 并投影到非负区域
            alpha(i) = max(0, alpha(i) + eta_t * grad_alpha);
        end
        w = w - eta_t * grad_w;
        b = b - eta_t * grad_b;
        % 每次迭代取最后的参数
        w_list = [w_list, w];
        b_list = [b_list, b];
        
        % 计算误差（hinge loss + regularization term）
        y_pre = svm_predict(w, b, X_test);
        acc(iter) = svm_report(y_test, y_pre);
    end
end
