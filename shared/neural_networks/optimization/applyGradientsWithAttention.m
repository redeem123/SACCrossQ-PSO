function network = applyGradientsWithAttention(network, dW_att, db_att, dW1, db1, dW2, db2, dW3, db3)
    % Apply gradients using Adam optimizer for network with attention
    %  Includes attention parameters in addition to standard parameters

    % Increment timestep
    network.t = network.t + 1;
    t = network.t;

    % Learning rate schedule with warmup
    if t <= network.warmupSteps
        lr = network.baseLearningRate * (t / network.warmupSteps);
    else
        lr = network.baseLearningRate;
    end

    network.learningRate = lr;

    % Adam optimizer parameters
    beta1 = network.beta1;
    beta2 = network.beta2;
    epsilon = network.epsilon;

    % === Update attention parameters ===
    % W_attention
    network.mW_attention = beta1 * network.mW_attention + (1 - beta1) * dW_att;
    network.vW_attention = beta2 * network.vW_attention + (1 - beta2) * (dW_att .^ 2);
    mW_att_hat = network.mW_attention / (1 - beta1^t);
    vW_att_hat = network.vW_attention / (1 - beta2^t);
    network.W_attention = network.W_attention - lr * mW_att_hat ./ (sqrt(vW_att_hat) + epsilon);

    % b_attention
    network.mb_attention = beta1 * network.mb_attention + (1 - beta1) * db_att;
    network.vb_attention = beta2 * network.vb_attention + (1 - beta2) * (db_att .^ 2);
    mb_att_hat = network.mb_attention / (1 - beta1^t);
    vb_att_hat = network.vb_attention / (1 - beta2^t);
    network.b_attention = network.b_attention - lr * mb_att_hat ./ (sqrt(vb_att_hat) + epsilon);

    % === Update W1, b1 ===
    network.mW1 = beta1 * network.mW1 + (1 - beta1) * dW1;
    network.vW1 = beta2 * network.vW1 + (1 - beta2) * (dW1 .^ 2);
    mW1_hat = network.mW1 / (1 - beta1^t);
    vW1_hat = network.vW1 / (1 - beta2^t);
    network.W1 = network.W1 - lr * mW1_hat ./ (sqrt(vW1_hat) + epsilon);

    network.mb1 = beta1 * network.mb1 + (1 - beta1) * db1;
    network.vb1 = beta2 * network.vb1 + (1 - beta2) * (db1 .^ 2);
    mb1_hat = network.mb1 / (1 - beta1^t);
    vb1_hat = network.vb1 / (1 - beta2^t);
    network.b1 = network.b1 - lr * mb1_hat ./ (sqrt(vb1_hat) + epsilon);

    % === Update W2, b2 ===
    network.mW2 = beta1 * network.mW2 + (1 - beta1) * dW2;
    network.vW2 = beta2 * network.vW2 + (1 - beta2) * (dW2 .^ 2);
    mW2_hat = network.mW2 / (1 - beta1^t);
    vW2_hat = network.vW2 / (1 - beta2^t);
    network.W2 = network.W2 - lr * mW2_hat ./ (sqrt(vW2_hat) + epsilon);

    network.mb2 = beta1 * network.mb2 + (1 - beta1) * db2;
    network.vb2 = beta2 * network.vb2 + (1 - beta2) * (db2 .^ 2);
    mb2_hat = network.mb2 / (1 - beta1^t);
    vb2_hat = network.vb2 / (1 - beta2^t);
    network.b2 = network.b2 - lr * mb2_hat ./ (sqrt(vb2_hat) + epsilon);

    % === Update W3, b3 ===
    network.mW3 = beta1 * network.mW3 + (1 - beta1) * dW3;
    network.vW3 = beta2 * network.vW3 + (1 - beta2) * (dW3 .^ 2);
    mW3_hat = network.mW3 / (1 - beta1^t);
    vW3_hat = network.vW3 / (1 - beta2^t);
    network.W3 = network.W3 - lr * mW3_hat ./ (sqrt(vW3_hat) + epsilon);

    network.mb3 = beta1 * network.mb3 + (1 - beta1) * db3;
    network.vb3 = beta2 * network.vb3 + (1 - beta2) * (db3 .^ 2);
    mb3_hat = network.mb3 / (1 - beta1^t);
    vb3_hat = network.vb3 / (1 - beta2^t);
    network.b3 = network.b3 - lr * mb3_hat ./ (sqrt(vb3_hat) + epsilon);
end
