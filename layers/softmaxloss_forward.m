function L = softmaxloss_forward(x, labels)
    % Inputs:
    %    x - Features. See the reshape command below. It is reshaped as for
    %        the fully connected layer.
    %    y - Labels. It is a vector with the correct labels. For
    %        instance if we have a batch of two where the first example is
    %        class 4 and the second example is class 7, labels is [4 7].
    %
    % Outputs:
    %    L - The computed loss. You should average over the batch elements.
    %        The loss for a single example is given is the assignment. Compute
    %        that value for all elements in the batch and then average.
    sz = size(x);
    batch = sz(end);
    features = prod(sz(1:end-1));

    assert(batch == numel(labels), 'Wrong number of labels given');
    % We reshape x in the same way as for the fully connected layer
    x = reshape(x, [features, batch]);
    % for numerical reasons. Convince yourself that the result is the same.
    x = bsxfun(@minus, x, min(x, [], 1));
    
    x_1 = arrayfun(@(k) exp(k),x);
    x_1 = log(sum(x_1));
    lables_2 = ones(1,batch);
    for i = 1:batch  
        labels_2(i) = sub2ind(sz,labels(i),i);
    end
    L = bsxfun(@(k,j) k - x(j)  ,x_1,labels_2);
    L = mean(L);
end