function [w, b, acc] = svm_gd(X, y, X_test, y_test, lambda, t, num_iters)
    % X: 输入数据矩阵，每行是一个样本
    % y: 标签向量，元素为 -1 或 1
    % lambda: 正则化参数
    % t: 步长
    % num_iters: 迭代次数

    [m, n] = size(X);
    w = zeros(n, 1);
    b = 0;
    w_list = [];
    b_list = [];
    acc = zeros(num_iters, 1); % 用于记录每次迭代的误差

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
        w = w - t * w_grad;
        b = b - t * b_grad;
        w_list = [w_list, w];
        b_list = [b_list, b];
        
        y_pre = svm_predict(w, b, X_test);
        acc(iter) = svm_report(y_test, y_pre);
    end
end
