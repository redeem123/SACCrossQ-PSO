function focalNet = trainFocalNetwork(focalNet, data, labels, epochs)
    % Train Focal Network using backpropagation (paper methodology)
    % Section 3.2.2: "the parameters in A are continuously updated through
    % the backpropagation algorithm"
    % Both MLP weights AND attention vector are trained together

    learningRate = 0.01;
    numSamples = size(data, 1);

    for epoch = 1:epochs
        % Apply attention to data (Paper Equation 5)
        attentionWeights = 1 ./ (1 + exp(-focalNet.attention)); % Sigmoid (Paper Eq. 4)
        weightedData = data .* attentionWeights'; % Hadamard product (Paper Eq. 5)

        % Forward pass through MLP
        z1 = weightedData * focalNet.W1' + focalNet.b1';
        a1 = max(0, z1); % ReLU activation

        z2 = a1 * focalNet.W2' + focalNet.b2';
        a2 = max(0, z2); % ReLU activation

        z3 = a2 * focalNet.W3' + focalNet.b3;
        predictions = 1 ./ (1 + exp(-z3)); % Sigmoid output

        % Compute loss (Binary Cross-Entropy)
        predictions = max(predictions, 1e-7);
        predictions = min(predictions, 1-1e-7);
        loss = -mean(labels .* log(predictions) + (1-labels) .* log(1-predictions));

        % Backpropagation through MLP
        % Output layer gradient
        dL_dz3 = (predictions - labels) / numSamples; % BCE gradient

        % Layer 3 gradients
        dL_dW3 = dL_dz3' * a2;
        dL_db3 = sum(dL_dz3);

        % Layer 2 gradients
        dL_da2 = dL_dz3 * focalNet.W3;
        dL_dz2 = dL_da2 .* (z2 > 0); % ReLU derivative
        dL_dW2 = dL_dz2' * a1;
        dL_db2 = sum(dL_dz2, 1)';

        % Layer 1 gradients
        dL_da1 = dL_dz2 * focalNet.W2;
        dL_dz1 = dL_da1 .* (z1 > 0); % ReLU derivative
        dL_dW1 = dL_dz1' * weightedData;
        dL_db1 = sum(dL_dz1, 1)';

        % Attention vector gradient (backprop through Hadamard product and sigmoid)
        dL_dWeightedData = dL_dz1 * focalNet.W1;
        % d(X * A*) / dA* = X (element-wise)
        dL_dAttentionWeights = sum(data .* dL_dWeightedData, 1)';
        % d(sigmoid(a)) / da = sigmoid(a) * (1 - sigmoid(a))
        dL_dAttention = dL_dAttentionWeights .* attentionWeights .* (1 - attentionWeights);

        % Update MLP weights with gradient descent
        focalNet.W3 = focalNet.W3 - learningRate * dL_dW3;
        focalNet.b3 = focalNet.b3 - learningRate * dL_db3;
        focalNet.W2 = focalNet.W2 - learningRate * dL_dW2;
        focalNet.b2 = focalNet.b2 - learningRate * dL_db2;
        focalNet.W1 = focalNet.W1 - learningRate * dL_dW1;
        focalNet.b1 = focalNet.b1 - learningRate * dL_db1;

        % Update attention vector
        focalNet.attention = focalNet.attention - learningRate * dL_dAttention;

        % Clamp attention to reasonable range to prevent numerical instability
        focalNet.attention = max(focalNet.attention, -5);
        focalNet.attention = min(focalNet.attention, 5);
    end
end

