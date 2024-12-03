function [w, b] = svm_subgradient_descent(X, y, lambda, learning_rate, num_iters)
    % X: 输入数据矩阵，每行是一个样本
    % y: 标签向量，元素为 -1 或 1
    % lambda: 正则化参数
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

% 示例数据
X = [1, 2; 2, 3; 3, 3; 2, 1; 3, 2];
y = [1; 1; 1; -1; -1];

% 超参数
lambda = 0.01;
learning_rate = 0.01;
num_iters = 1000;

% 训练 SVM
[w, b] = svm_subgradient_descent(X, y, lambda, learning_rate, num_iters);

% 显示结果
disp('权重向量 w:');
disp(w);
disp('偏置 b:');
disp(b);
