function plot_svm_decision_boundary(X, y, w, b)
% X: data matrix, with each row being a sample
% y: label vector with elements -1 or 1
% w: the trained weight vector
% b: the trained bias vector

% plot data points
figure;
hold on;
pos = find(y == 1);
neg = find(y == -1);
scatter(X(pos, 1), X(pos, 2), 'ro', 'filled'); % 正类
scatter(X(neg, 1), X(neg, 2), 'bo', 'filled'); % 负类

% plot hyperplain wx + b = 0
x_min = min(X(:, 1));
x_max = max(X(:, 1));
x_plot = linspace(x_min, x_max, 100);
y_plot = -(w(1) * x_plot + b) / w(2);
plot(x_plot, y_plot, 'k-', 'LineWidth', 2); % 决策边界

% set plot parameters
xlabel('Feature 1');
ylabel('Feature 2');
title('SVM Decision Boundary');
legend('Positive Class', 'Negative Class', 'Decision Boundary');
grid on;
hold off;

end
