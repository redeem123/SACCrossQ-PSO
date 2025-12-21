function dy = leakyReLU_gradient(x, alpha)
    % Gradient of Leaky ReLU activation function
    % Used by RL-NNPSO TD3 networks for backpropagation
    %
    % Inputs:
    %   x: Input values (pre-activation)
    %   alpha: Slope for negative values (default: 0.01)
    %
    % Output:
    %   dy: Gradient values

    if nargin < 2
        alpha = 0.01;  % Default value for backward compatibility
    end

    dy = ones(size(x));
    dy(x < 0) = alpha;
end
