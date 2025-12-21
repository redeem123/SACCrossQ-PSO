function output = leakyReLU(x, alpha)
    % Leaky ReLU activation function
    % Used by RL-NNPSO TD3 networks
    %
    % Inputs:
    %   x: Input values
    %   alpha: Slope for negative values (default: 0.01)
    %
    % Output:
    %   output: Activated values

    if nargin < 2
        alpha = 0.01;  % Default value for backward compatibility
    end

    output = max(alpha * x, x);
end

