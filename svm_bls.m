function [w, b, iter, w_list, b_list, acc] = svm_bls(X, y, X_test, y_test, lambda, num_iters, alpha, beta, t0)
    % X: 输入数据矩阵，每行是一个样本
    % y: 标签向量，元素为 -1 或 1
    % lambda: 正则化参数
    % num_iters: 最大迭代次数
    % alpha: Backtracking Line Search 参数，0 < alpha < 1/2
    % beta: Backtracking Line Search 参数，0 < beta < 1
    % t0: Backtracking Line Search 截止条件

    [m, n] = size(X);
    w = zeros(n, 1); % 初始化权重
    b = 0;           % 初始化偏置
    w_list = [];     % 保存每次完整迭代后的权重
    b_list = [];     % 保存每次完整迭代后的偏置
    acc = zeros(num_iters, 1); % 保存每次迭代后的准确率

    for iter = 1:num_iters
        prev_w = w; % 保存上一代的 w
        prev_b = b; % 保存上一代的 b

        % 遍历所有样本
        for i = 1:m
            xi = X(i, :)';
            yi = y(i);

            % 计算当前损失
            loss = lambda * 0.5 * (w' * w) + max(0, 1 - yi * (w' * xi + b));
            
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

            % Backtracking Line Search
            t = 1;
            while true
                % 更新参数
                w_new = w - t * w_grad;
                b_new = b - t * b_grad;
                
                % 计算新的损失
                new_loss = lambda * 0.5 * (w_new' * w_new) + max(0, 1 - yi * (w_new' * xi + b_new));
                
                % 检查条件
                if new_loss <= loss - alpha * t * (norm(w_grad)^2 + b_grad^2)
                    break;
                end
                
                % 减小步长
                t = beta * t;
                if t < t0
                    % 如果步长过小，填充剩余的 acc 为最近有效值
                    y_pre = svm_predict(w, b, X_test);
                    acc_0 = svm_report(y_test, y_pre);
                    acc(iter:end) = acc_0;
                    return;
                end
            end

            % 更新权重和偏置
            w = w_new;
            b = b_new;
        end

        % 在每次完整遍历后保存最新的 w 和 b
        w_list = [w_list, w];
        b_list = [b_list, b];

        % 使用最新 w 和 b 计算测试集准确率
        y_pre = svm_predict(w, b, X_test);
        acc(iter) = svm_report(y_test, y_pre);
    end
end