function [w, b, total_iter] = svm_barrier(X, y, t, mu, tol, num_iters, step_size)
% X: 输入数据矩阵，每行是一个样本
% y: 标签向量，元素为 -1 或 1
% t: 
% mu: 
% tol: 
% num_iters: 最大迭代次数
% step_size: 步长

% 初始化参数
[m, n] = size(X);
w = zeros(n, 1);
b = 0;
total_iter = 0;

% 优化过程
for iter = 1:num_iters
    % 使用梯度下降法进行优化中心步骤
    disp(w);
    disp(b);
    [w, b, func_iter] = svm_centering(X, y, w, b, t, num_iters, step_size, 1e-5);
    disp(w);
    disp(b);
    plot_svm_decision_boundary(X, y, w, b);
    disp('------------');
    total_iter = total_iter + func_iter;

    % 检查收敛
    if m / t < tol
        break;
    else
        t = mu * t;
    end
end

end
