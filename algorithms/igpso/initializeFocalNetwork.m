function focalNet = initializeFocalNetwork(numFeatures, hiddenNeurons)
    % Initialize Focal Neural Network with attention mechanism
    
    focalNet = struct();
    
    % Attention vector (all initialized to 1 as per paper)
    focalNet.attention = ones(numFeatures, 1);
    
    % MLP with two hidden layers as per paper Figure 1
    focalNet.W1 = randn(hiddenNeurons, numFeatures) * sqrt(2 / numFeatures);
    focalNet.b1 = zeros(hiddenNeurons, 1);
    
    focalNet.W2 = randn(hiddenNeurons, hiddenNeurons) * sqrt(2 / hiddenNeurons);
    focalNet.b2 = zeros(hiddenNeurons, 1);
    
    focalNet.W3 = randn(1, hiddenNeurons) * sqrt(2 / hiddenNeurons);
    focalNet.b3 = 0;
end

