function ann = initializeConsistentANN(inputSize, hiddenNeurons)
    % Initialize ANN exactly as specified in paper
    % PAPER: anni â†? random neural network, fitness(anni) â†? 0
    
    ann = struct();
    ann.inputSize = inputSize;      % Problem dimensionality 
    ann.hiddenSize = hiddenNeurons; % Hidden layer size
    ann.outputSize = 1;             % Single output (PAPER SPECIFICATION)
    ann.success = 0;                % PAPER: fitness(anni) â†? 0
    
    % Random weight initialization in [-1, 1] as mentioned in paper
    ann.W1 = -1 + 2 * rand(hiddenNeurons, inputSize);  % Hidden layer weights
    ann.b1 = -1 + 2 * rand(hiddenNeurons, 1);          % Hidden layer biases
    
    ann.W2 = -1 + 2 * rand(1, hiddenNeurons);          % Output layer weights  
    ann.b2 = -1 + 2 * rand(1, 1);                      % Output layer bias
end

