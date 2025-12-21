function clipped = clipByNorm(grad, maxNorm)
    % Clip gradients by norm
    % Used by RL-NNPSO TD3 networks for gradient clipping
    %
    % Inputs:
    %   grad: Gradient matrix/vector
    %   maxNorm: Maximum allowed norm
    %
    % Output:
    %   clipped: Clipped gradient

    gradNorm = norm(grad(:));
    if gradNorm > maxNorm
        clipped = grad * (maxNorm / gradNorm);
    else
        clipped = grad;
    end
end
