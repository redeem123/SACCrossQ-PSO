function [dW_att, db_att, dW1, db1, dW2, db2, dW3, db3] = computeGradientsWithAttention(network, inputs, targets, h1, h2, outputs, attentionWeights, attentionOutput)
    % Compute gradients for network with feature attention
    % Uses backpropagation through attention mechanism

    batchSize = size(inputs, 2);

    % === Output layer gradients ===
    dOutput = outputs - targets;  % [outputSize, batchSize]

    dW3 = (dOutput * h2') / batchSize;
    db3 = mean(dOutput, 2);

    % === Hidden layer 2 gradients ===
    dh2 = network.W3' * dOutput;  % [hiddenSize2, batchSize]
    dz2 = dh2 .* (h2 > 0);  % ReLU derivative

    dW2 = (dz2 * h1') / batchSize;
    db2 = mean(dz2, 2);

    % === Hidden layer 1 gradients ===
    dh1 = network.W2' * dz2;  % [hiddenSize1, batchSize]
    dz1 = dh1 .* (h1 > 0);  % ReLU derivative

    dW1 = (dz1 * attentionOutput') / batchSize;
    db1 = mean(dz1, 2);

    % === Attention layer gradients ===
    % Gradient w.r.t. attention output
    dAttentionOutput = network.W1' * dz1;  % [inputSize, batchSize]

    % Gradient through element-wise multiplication: attentionOutput = attentionWeights .* inputs
    dAttentionWeights = dAttentionOutput .* inputs;  % [inputSize, batchSize]
    dInputs_fromAttention = dAttentionOutput .* attentionWeights;  % We don't use this (inputs are fixed)

    % Gradient through sigmoid: attentionWeights = sigmoid(attention_logits)
    % sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    dAttentionLogits = dAttentionWeights .* attentionWeights .* (1 - attentionWeights);  % [inputSize, batchSize]

    % Gradient w.r.t. attention parameters
    dW_att = (dAttentionLogits * inputs') / batchSize;
    db_att = mean(dAttentionLogits, 2);
end
