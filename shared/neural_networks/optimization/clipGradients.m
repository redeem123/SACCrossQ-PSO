function clippedGrad = clipGradients(grad, maxNorm)
    % Clip gradients to prevent exploding gradients
    gradNorm = norm(grad(:));
    if gradNorm > maxNorm
        clippedGrad = grad * (maxNorm / gradNorm);
    else
        clippedGrad = grad;
    end
    
    % Ensure finite gradients
    clippedGrad(~isfinite(clippedGrad)) = 0;
end

