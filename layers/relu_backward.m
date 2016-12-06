function dldx = relu_backward(x, dldy)
    %y = arrayfun(@(x) max(x,0)/x,x);
    y=x;
    y(y<0) = 0;
    y(y>0) = 1;
    dldx = dldy.*y;

end
