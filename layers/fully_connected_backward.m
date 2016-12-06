function [dldX, dldA, dldb] = fully_connected_backward(X, dldY, A, b)
    % Inputs
    %   X - The input variable. The size might vary, but the last dimension
    %       tells you which element in the batch it is.
    %   dldy - The partial derivatives of the loss with respect to the
    %       output varible y. The size of dldy is the same as for y as
    %       computed in the forward pass.
    %   A  - The weight matrix
    %   b  - The bias vector
    %
    % Outputs
    %    dldX - Gradient backpropagated to X
    %    dldA - Gradient backpropagated to A
    %    dldb - Gradient backpropagated to b
    %
    % All gradients should have the same size as the variable. That is,
    % dldx and x should have the same size, dldA and A the same size and dldb
    % and b also the same size.
    sz_X = size(X);
    batch_X = sz_X(end);
    features_X = prod(sz_X(1:end-1));
    
    sz_A = size(A);
    batch_A = sz_A(end);
    features_A = prod(sz_A(1:end-1));
    
    sz_b = size(b);
    batch_b = sz_b(end);
    features_b = prod(sz_b(1:end-1));
    
    % We reshape the input vector so that all features for a single batch
    % element are in the columns. X is now as defined in the assignment.
    X = reshape(X, [features_X, batch_X]);
    A = reshape(A, [features_A, batch_A]);
    b = reshape(b, [features_b, batch_b]);
    
    assert(size(A, 2) == features_X, ...
        sprintf('Expected %d columns in the weights matrix, got %d', features_X, size(A,2)));
    assert(size(A, 1) == numel(b), 'Expected as many rows in A as elements in b');
    
    % Implement it here.
    dldX = A'*dldY;
    dldA = dldY*X';
    dldb = dldY * ones(1,size(dldY,2))';
    % note that dldX should have the same size as X, so use reshape
    dldX = reshape(dldX, sz_X);
    dldA = reshape(dldA, sz_A);
    dldb = reshape(dldb, sz_b);
end
