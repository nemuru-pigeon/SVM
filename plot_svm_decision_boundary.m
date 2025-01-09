function plot_svm_decision_boundary(X, y, w, b)
    % X: 输入数据矩阵，每行是一个样本
    % y: 标签向量，元素为 -1 或 1
    % w: 训练得到的权重向量
    % b: 训练得到的偏置

    % 绘制数据点
    figure;
    hold on;
    pos = find(y == 1);
    neg = find(y == -1);
    scatter(X(pos, 1), X(pos, 2), 'ro', 'filled'); % 正类
    scatter(X(neg, 1), X(neg, 2), 'bo', 'filled'); % 负类

    % 绘制决策边界 wx + b = 0
    x_min = min(X(:, 1));
    x_max = max(X(:, 1));
    x_plot = linspace(x_min, x_max, 100);
    y_plot = -(w(1) * x_plot + b) / w(2);
    plot(x_plot, y_plot, 'k-', 'LineWidth', 2); % 决策边界

    % 设置图形参数
    xlabel('Feature 1');
    ylabel('Feature 2');
    title('SVM Decision Boundary');
    legend('Positive Class', 'Negative Class', 'Decision Boundary');
    grid on;
    hold off;
end
