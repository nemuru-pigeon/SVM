function [w, b, iter] = svm_bls(X, y, lambda, num_iters, alpha, beta, t0)
    % X: 输入数据矩阵，每行是一个样本
    % y: 标签向量，元素为 -1 或 1
    % lambda: 正则化参数
    % num_iters: 最大迭代次数
    % alpha: Backtracking Line Search 参数，0 < a < 1/2
    % beta: Backtracking Line Search 参数，0 < b < 1
    % t0: Backtracking Line Search 截止条件

    [m, n] = size(X);
    w = zeros(n, 1);
    b = 0;

    for iter = 1:num_iters
        % 遍历所有样本
        for i = 1:m
            xi = X(i, :)';
            yi = y(i);

            % 计算损失函数值
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

            % backtracking line search
            t = 1;
            while true
                % 更新参数
                w_new = w - t * w_grad;
                b_new = b - t * b_grad;
                
                % 计算新的损失
                new_loss = lambda * 0.5 * (w_new' * w_new) + max(0, 1 - yi * (w_new' * xi + b_new));
                
                % 检查条件
                if new_loss < loss - alpha * t * (w_grad' * w_grad + b_grad^2)
                    break;
                end
                
                % 减小步长
                t = beta * t;
                if t < t0
                    disp(t);
                    return;
                end
            end

            % 更新权重和截距
            w = w_new;
            b = b_new;
        end
    end
end
