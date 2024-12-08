function [y_pre] = svm_predict(w, b, X)
% w: the weight of the classifier
% b: the bias of the classifier
% X: test data

y = X * w + b;
y_pre = sign(y);

end

