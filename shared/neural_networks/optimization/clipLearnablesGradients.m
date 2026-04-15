function clippedGrads = clipLearnablesGradients(gradients, maxNorm)
    % Clip dlnetwork Learnables gradients by global L2 norm.
    % Zeros out NaN/Inf entries before computing the norm.
    %
    % Inputs:
    %   gradients: Table of gradients from dlfeval (dlnetwork.Learnables format)
    %   maxNorm:   Maximum allowed gradient norm
    %
    % Returns:
    %   clippedGrads: Gradients clipped to maxNorm

    totalNorm = 0;
    for i = 1:height(gradients)
        g = gradients.Value{i};
        badMask = isnan(g) | isinf(g);
        if any(badMask(:))
            g(badMask) = 0;
            gradients.Value{i} = g;
        end
        totalNorm = totalNorm + sum(g(:).^2);
    end
    totalNorm = sqrt(totalNorm);

    if totalNorm > maxNorm
        c = maxNorm / (totalNorm + 1e-6);
        for i = 1:height(gradients)
            gradients.Value{i} = gradients.Value{i} * c;
        end
    end

    clippedGrads = gradients;
end
