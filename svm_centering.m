function [w, b, iter] = svm_centering(X, y, w, b, t, num_iters, step_size, tol)
% X: 输入数据矩阵，每行是一个样本
% y: 标签向量，元素为 -1 或 1
% w
% b
% t: 
% num_iters: 最大迭代次数
% step_size: 步长
% tol: 

[m, n] = size(X);
step_size = 0.01;

alpha = 0.25;
beta = 0.5;
step_size0 = 1e-5;

old_w = zeros(n, 1);
old_b = 0;

for iter = 1:num_iters
    w_grad = t * w;
    b_grad = 0;
    loss = t * 0.5 * (w' * w);

    % 遍历所有样本
    for i = 1:m
        xi = X(i, :)';
        yi = y(i);

        % 计算损失函数值
        loss = loss - log(yi * (w' * xi + b) - 1);
        
        % 计算梯度
        w_grad = w_grad - yi * xi / (yi * (w' * xi + b) - 1);
        b_grad = b_grad + yi / (1 - yi * (w' * xi + b));
    end

    % backtracking line search
    while true
        % 更新参数
        w_new = w - step_size * w_grad;
        b_new = b - step_size * b_grad;
        
        % 计算新的损失
        new_loss = t * 0.5 * (w_new' * w_new);
        for i = 1:m
            xi = X(i, :)';
            yi = y(i);
    
            % 计算损失函数值
            new_loss = new_loss - log(yi * (w_new' * xi + b_new) - 1);
        end
        
        % 检查条件
        if new_loss < loss - alpha * step_size * (w_grad' * w_grad + b_grad^2)
            break;
        end
        
        % 减小步长
        step_size = beta * step_size;
        if step_size < step_size0
            return;
        end
    end

    % 更新权重和截距
    w = w_new;
    b = b_new;
    
    % % 更新参数
    % w = w - step_size * w_grad;
    % b = b - step_size * b_grad;

    if norm(old_w - w) < tol && abs(old_b - b) < tol
        return;
    end
    old_w = w;
    old_b = b;
end

end
