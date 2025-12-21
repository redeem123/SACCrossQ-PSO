function output = forwardPassConsistentANN(ann, input)
    % Forward pass exactly as specified in paper
    % Simple MLP with one hidden layer and single output
    
    % Validate input size
    if length(input) ~= ann.inputSize
        error('Input size mismatch: expected %d, got %d', ann.inputSize, length(input));
    end
    
    % Hidden layer with sigmoid activation (common in 2020 for such applications)
    z1 = ann.W1 * input + ann.b1;
    h1 = 1 ./ (1 + exp(-z1));  % Sigmoid activation
    
    % Output layer with linear activation (single response value)
    output = ann.W2 * h1 + ann.b2;
    
    % Ensure scalar output
    output = output(1);
end

