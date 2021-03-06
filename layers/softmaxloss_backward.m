function dldx = softmaxloss_backward(x, labels)
    % Inputs:
    %    x - Features. See the reshape command below. It is reshaped as for
    %        the fully connected layer.
    %    y - Labels. It is a vector with the correct labels. For
    %        instance if we have a batch of two where the first example is
    %        class 4 and the second example is class 7, labels is [4 7].
    %
    % Outputs:
    %    dldx - Partial derivative of L with respect to x. Remember that in
    %           the forward pass you average over the batch elements.
    sz = size(x);
    batch = sz(end);
    features = prod(sz(1:end-1));
    labels = labels(:);
    % suitable for matrix multiplication
    x = reshape(x, [features, batch]);
    % for numerical stability. Convince yourself that the result is the same.
    x = bsxfun(@minus, x, min(x, [], 1));
    
    x_1 = exp(x);
    x_2 = sum(x_1);

    dldx = bsxfun(@rdivide,x_1,x_2);
    
    dldx(sub2ind(size(dldx),labels',1:batch)) =  (-1) + ...
        dldx(sub2ind(size(dldx),labels',1:batch));
    dldx = dldx/batch;
    dldx = reshape(dldx, sz);

end
