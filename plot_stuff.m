    % Plot first 36 wrongly classified digits 
    if true
        figure();
        pred_err = pred;
        pred_err(vec(pred) == vec(y_test)) = [];
        y_err = y_test;
        y_err(vec(pred) == vec(y_test)) = [];
        x_err = x_test;
        x_err(:,:,:,vec(pred) == vec(y_test)) = [];
        
        for i=1:min(length(pred_err),6)
            for j=1:min(length(pred_err),6)
                subplot(6,6,6*(i-1)+j);
                imagesc(x_err(:,:,:,6*(i-1)+j));
                colormap(gray);
                title(['C: ' num2str(pred_err(6*(i-1)+j)) ', G: ' num2str(y_err(6*(i-1)+j))]);
                axis off;
            end
        end
    end
    
        % Plot filers
    if true
        figure();
        title('Filters for fist convolution layer');
        for i=1:length(net.layers)
            if strcmpi(net.layers{i}.type, 'convolution'),
                for j = 1:size(net.layers{i}.params.weights,4),
                    subplot(4,4,j);
                    imagesc(net.layers{i}.params.weights(:,:,1,j));
                    colormap(gray);
                    axis off;
                end
                break;
            end
        end
    end