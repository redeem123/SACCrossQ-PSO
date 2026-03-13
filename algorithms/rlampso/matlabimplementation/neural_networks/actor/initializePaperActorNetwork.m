function actor = initializePaperActorNetwork(stateSize, actionSize)
    % PAPER Figure 4: 4-layer fully connected + LeakyReLU + tanh output
    
    actor = struct();
    
    % PAPER architecture: Input -> 64 -> 64 -> 64 -> output
    hiddenSize = 64;
    
    % Xavier initialization
    scale1 = sqrt(2 / stateSize);
    scale2 = sqrt(2 / hiddenSize);
    
    % Layer 1: Input -> Hidden1 (64)
    actor.W1 = randn(hiddenSize, stateSize) * scale1;
    actor.b1 = zeros(hiddenSize, 1);
    
    % Layer 2: Hidden1 -> Hidden2 (64)
    actor.W2 = randn(hiddenSize, hiddenSize) * scale2;
    actor.b2 = zeros(hiddenSize, 1);
    
    % Layer 3: Hidden2 -> Hidden3 (64)
    actor.W3 = randn(hiddenSize, hiddenSize) * scale2;
    actor.b3 = zeros(hiddenSize, 1);
    
    % Output Layer: Hidden3 -> Output (20 for traditional PSO)
    actor.W4 = randn(actionSize, hiddenSize) * scale2;
    actor.b4 = zeros(actionSize, 1);
end

