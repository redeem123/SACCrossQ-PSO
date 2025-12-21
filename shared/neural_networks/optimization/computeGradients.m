function [dW1, db1, dW2, db2, dW3, db3] = computeGradients(network, input, target, h1, h2, output)
    % Compute proper gradients using backpropagation
    
    batchSize = size(input, 2);
    if batchSize == 0
        error('Empty batch provided for gradient computation');
    end
    
    % Ensure target and output have same dimensions
    if size(target) ~= size(output)
        error('Target and output dimension mismatch');
    end
    
    % Output layer gradients (MSE loss: 0.5 * ||output - target||^2)
    dLoss_dOutput = (output - target) / batchSize;  % Gradient of MSE loss
    
    % Gradients w.r.t. output layer parameters
    dW3 = dLoss_dOutput * h2';  % [outputSize x hiddenSize2]
    db3 = sum(dLoss_dOutput, 2);  % [outputSize x 1]
    
    % Hidden layer 2 gradients
    dLoss_dH2 = network.W3' * dLoss_dOutput;  % [hiddenSize2 x batchSize]
    
    % ReLU derivative: gradient is 1 if input > 0, else 0
    dH2_dZ2 = double(h2 > 0);  % Element-wise ReLU derivative
    dLoss_dZ2 = dLoss_dH2 .* dH2_dZ2;  % [hiddenSize2 x batchSize]
    
    % Gradients w.r.t. hidden layer 2 parameters
    dW2 = dLoss_dZ2 * h1';  % [hiddenSize2 x hiddenSize1]
    db2 = sum(dLoss_dZ2, 2);  % [hiddenSize2 x 1]
    
    % Hidden layer 1 gradients
    dLoss_dH1 = network.W2' * dLoss_dZ2;  % [hiddenSize1 x batchSize]
    
    % ReLU derivative for hidden layer 1
    dH1_dZ1 = double(h1 > 0);  % Element-wise ReLU derivative
    dLoss_dZ1 = dLoss_dH1 .* dH1_dZ1;  % [hiddenSize1 x batchSize]
    
    % Gradients w.r.t. hidden layer 1 parameters
    dW1 = dLoss_dZ1 * input';  % [hiddenSize1 x inputSize]
    db1 = sum(dLoss_dZ1, 2);  % [hiddenSize1 x 1]
    
    % Gradient clipping to prevent exploding gradients
    maxGradNorm = 5.0;
    
    % Clip weight gradients
    dW1 = clipGradients(dW1, maxGradNorm);
    dW2 = clipGradients(dW2, maxGradNorm);
    dW3 = clipGradients(dW3, maxGradNorm);
    
    % Clip bias gradients
    db1 = clipGradients(db1, maxGradNorm);
    db2 = clipGradients(db2, maxGradNorm);
    db3 = clipGradients(db3, maxGradNorm);
end

