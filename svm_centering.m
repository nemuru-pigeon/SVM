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
% w = zeros(n, 1);
% b = 0;
old_w = zeros(n, 1);
old_b = 0;

for iter = 1:num_iters
    w_grad = t * w;
    b_grad = 0;

    % 遍历所有样本
    for i = 1:m
        xi = X(i, :)';
        yi = y(i);
        
        % 计算梯度
        w_grad = w_grad - yi * xi / (yi * (w' * xi + b) - 1);
        b_grad = b_grad + yi / (1 - yi * (w' * xi + b));
    end
    
    % 更新参数
    w = w - step_size * w_grad;
    b = b - step_size * b_grad;

    if norm(old_w - w) < tol && abs(old_b - b) < tol
        return;
    end
    old_w = w;
    old_b = b;
end

end
