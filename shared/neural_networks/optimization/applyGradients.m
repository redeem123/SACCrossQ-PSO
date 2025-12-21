function network = applyGradients(network, dW1, db1, dW2, db2, dW3, db3)
    % Apply gradients using Adam optimizer with proper momentum updates
    
    % Increment time step
    network.t = network.t + 1;
    
    % Learning rate schedule (optional)
    if network.t <= network.warmupSteps
        currentLR = network.baseLearningRate * (network.t / network.warmupSteps);
    else
        % Optional decay
        decay = 0.99;
        currentLR = network.baseLearningRate * (decay ^ floor(network.t / 100));
    end
    
    % Bias correction terms
    beta1_corrected = 1 - network.beta1^network.t;
    beta2_corrected = 1 - network.beta2^network.t;
    
    % Update Layer 1 weights
    network.mW1 = network.beta1 * network.mW1 + (1 - network.beta1) * dW1;
    network.vW1 = network.beta2 * network.vW1 + (1 - network.beta2) * (dW1.^2);
    
    mW1_corrected = network.mW1 / beta1_corrected;
    vW1_corrected = network.vW1 / beta2_corrected;
    
    network.W1 = network.W1 - currentLR * mW1_corrected ./ (sqrt(vW1_corrected) + network.epsilon);
    
    % Update Layer 1 biases
    network.mb1 = network.beta1 * network.mb1 + (1 - network.beta1) * db1;
    network.vb1 = network.beta2 * network.vb1 + (1 - network.beta2) * (db1.^2);
    
    mb1_corrected = network.mb1 / beta1_corrected;
    vb1_corrected = network.vb1 / beta2_corrected;
    
    network.b1 = network.b1 - currentLR * mb1_corrected ./ (sqrt(vb1_corrected) + network.epsilon);
    
    % Update Layer 2 weights
    network.mW2 = network.beta1 * network.mW2 + (1 - network.beta1) * dW2;
    network.vW2 = network.beta2 * network.vW2 + (1 - network.beta2) * (dW2.^2);
    
    mW2_corrected = network.mW2 / beta1_corrected;
    vW2_corrected = network.vW2 / beta2_corrected;
    
    network.W2 = network.W2 - currentLR * mW2_corrected ./ (sqrt(vW2_corrected) + network.epsilon);
    
    % Update Layer 2 biases
    network.mb2 = network.beta1 * network.mb2 + (1 - network.beta1) * db2;
    network.vb2 = network.beta2 * network.vb2 + (1 - network.beta2) * (db2.^2);
    
    mb2_corrected = network.mb2 / beta1_corrected;
    vb2_corrected = network.vb2 / beta2_corrected;
    
    network.b2 = network.b2 - currentLR * mb2_corrected ./ (sqrt(vb2_corrected) + network.epsilon);
    
    % Update Layer 3 weights
    network.mW3 = network.beta1 * network.mW3 + (1 - network.beta1) * dW3;
    network.vW3 = network.beta2 * network.vW3 + (1 - network.beta2) * (dW3.^2);
    
    mW3_corrected = network.mW3 / beta1_corrected;
    vW3_corrected = network.vW3 / beta2_corrected;
    
    network.W3 = network.W3 - currentLR * mW3_corrected ./ (sqrt(vW3_corrected) + network.epsilon);
    
    % Update Layer 3 biases
    network.mb3 = network.beta1 * network.mb3 + (1 - network.beta1) * db3;
    network.vb3 = network.beta2 * network.vb3 + (1 - network.beta2) * (db3.^2);
    
    mb3_corrected = network.mb3 / beta1_corrected;
    vb3_corrected = network.vb3 / beta2_corrected;
    
    network.b3 = network.b3 - currentLR * mb3_corrected ./ (sqrt(vb3_corrected) + network.epsilon);
    
    % Weight decay (L2 regularization)
    lambda = 1e-5;
    network.W1 = network.W1 * (1 - currentLR * lambda);
    network.W2 = network.W2 * (1 - currentLR * lambda);
    network.W3 = network.W3 * (1 - currentLR * lambda);
    
    % Ensure all parameters remain finite
    network.W1(~isfinite(network.W1)) = 0;
    network.W2(~isfinite(network.W2)) = 0;
    network.W3(~isfinite(network.W3)) = 0;
    network.b1(~isfinite(network.b1)) = 0;
    network.b2(~isfinite(network.b2)) = 0;
    network.b3(~isfinite(network.b3)) = 0;
end

