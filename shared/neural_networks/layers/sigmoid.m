function y = sigmoid(x)
    % Sigmoid activation function with numerical stability
    y = 1 ./ (1 + exp(-max(-500, min(500, x))));
end

