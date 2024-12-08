function [accuracy] = svm_report(y_test, y_pre)
% y_test: the true labels
% y_pre: the predicted labels

num = size(y_test, 1);

accurate_num = 0;
for i = 1 : num
    if y_test(i) == y_pre(i)
        accurate_num = accurate_num + 1;
    end
end

accuracy = accurate_num / num;

end

