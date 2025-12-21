function eliteNetwork = updateNormalizationStats(eliteNetwork, inputs)
    % Update running statistics for input normalization
    
    batchMean = mean(inputs, 2);
    batchStd = std(inputs, 0, 2);
    
    % Replace NaN/Inf with safe values
    batchMean(~isfinite(batchMean)) = 0;
    batchStd(~isfinite(batchStd)) = 1;
    batchStd(batchStd < 0.1) = 1; % Prevent division by very small numbers
    
    % Running average update
    alpha = 0.1;
    eliteNetwork.inputMean = (1 - alpha) * eliteNetwork.inputMean + alpha * batchMean;
    eliteNetwork.inputStd = (1 - alpha) * eliteNetwork.inputStd + alpha * batchStd;
    
    % Ensure safe values
    eliteNetwork.inputMean(~isfinite(eliteNetwork.inputMean)) = 0;
    eliteNetwork.inputStd(~isfinite(eliteNetwork.inputStd)) = 1;
    eliteNetwork.inputStd(eliteNetwork.inputStd < 0.1) = 1;
end

