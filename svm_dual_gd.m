function [w, b, alpha, acc] = svm_dual_gd(X, y, X_test, y_test, eta, num_iters)
    % 硬间隔 SVM 对偶问题的投影梯度法实现
    % X: 输入数据矩阵，每行是一个样本
    % y: 标签向量 (-1 或 1)
    % X_test: 测试集数据
    % y_test: 测试集标签
    % eta: 初始学习率
    % num_iters: 最大迭代次数

    [m, n] = size(X);        % 样本数量 m 和特征数量 n
    alpha = zeros(m, 1);     % 初始化拉格朗日乘子
    acc = zeros(num_iters, 1); % 存储测试集准确率

    for iter = 1:num_iters
        % 学习率衰减
        eta_t = eta / (1 + iter / 10);

        % 对每个 alpha_i 更新
        for i = 1:m
            % 计算梯度
            grad = 1 - y(i) * sum(alpha .* y .* (X * X(i, :)'));

            % 梯度更新
            alpha(i) = alpha(i) + eta_t * grad;

            % 投影到 [0, +∞)
            alpha(i) = max(0, alpha(i));
        end

        % 修正以满足 \sum alpha_i y_i = 0
        delta = sum(alpha .* y);
        alpha = alpha - y * (delta / sum(y.^2));

        % 更新 w
        w = sum((alpha .* y) .* X, 1)'; % w = sum(alpha_i * y_i * x_i)

        % 更新偏置 b
        support_indices = find(alpha > 1e-5); % 支持向量
        if ~isempty(support_indices)
            b = mean(y(support_indices) - X(support_indices, :) * w);
        else
            b = 0; % 理论上不应该发生
        end

        % 预测测试集
        y_pre = svm_predict(w, b, X_test);

        % 记录测试集准确率
        acc(iter) = svm_report(y_test, y_pre);
    end
end