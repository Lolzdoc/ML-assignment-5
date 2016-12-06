function y = relu_forward(x)
    y = x;
    y(y<0) = 0;
end
