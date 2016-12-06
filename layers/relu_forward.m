function y = relu_forward(x)
y = x;
y(y<0) = 0;
% y = arrayfun(@(x) max(x,0),x);
end
