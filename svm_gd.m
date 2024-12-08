function [w, b] = svm_gd(X, y, lambda, learning_rate, num_iters)
    % X: 输入数据矩阵，每行是一个样本
    % y: 标签向量，元素为 -1 或 1
    % lambda: 正则化参数，对w部分用或对约束部分用
    % learning_rate: 学习率
    % num_iters: 迭代次数

    [m, n] = size(X);
    w = zeros(n, 1);
    b = 0;

    for iter = 1:num_iters
        % 随机选择一个样本
        idx = randi(m);
        xi = X(idx, :)';
        yi = y(idx);
        
        % 检查是否满足条件
        condition = yi * (w' * xi + b) < 1;
        
        % 计算梯度
        if condition
            w_grad = lambda * w - yi * xi;
            b_grad = -yi;
        else
            w_grad = lambda * w;
            b_grad = 0;
        end
        
        % 更新参数
        w = w - learning_rate * w_grad;
        b = b - learning_rate * b_grad;
    end
end
