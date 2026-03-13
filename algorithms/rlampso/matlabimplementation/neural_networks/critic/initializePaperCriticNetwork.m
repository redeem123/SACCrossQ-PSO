function critic = initializePaperCriticNetwork(stateSize, actionSize)
    % PAPER Figure 5: 6-layer fully connected + LeakyReLU
    
    critic = struct();
    
    inputSize = stateSize + actionSize; % 15 + 20 = 35
    
    % PAPER architecture: Input -> 64 -> 64 -> 32 -> 32 -> 16 -> 1
    critic.W1 = randn(64, inputSize) * sqrt(2 / inputSize);
    critic.b1 = zeros(64, 1);
    
    critic.W2 = randn(64, 64) * sqrt(2 / 64);
    critic.b2 = zeros(64, 1);
    
    critic.W3 = randn(32, 64) * sqrt(2 / 64);
    critic.b3 = zeros(32, 1);
    
    critic.W4 = randn(32, 32) * sqrt(2 / 32);
    critic.b4 = zeros(32, 1);
    
    critic.W5 = randn(16, 32) * sqrt(2 / 32);
    critic.b5 = zeros(16, 1);
    
    critic.W6 = randn(1, 16) * sqrt(2 / 16);
    critic.b6 = zeros(1, 1);
end

