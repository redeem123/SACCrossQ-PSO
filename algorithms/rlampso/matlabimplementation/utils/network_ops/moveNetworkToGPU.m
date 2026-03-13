function network = moveNetworkToGPU(network)
    % Move all network weights and biases to GPU
    %
    % Inputs:
    %   network: Struct containing network weights (W1, b1, W2, b2, etc.)
    %
    % Returns:
    %   network: Same struct with weights moved to GPU

    if ~canUseGPU()
        return;  % GPU not available
    end

    fields = fieldnames(network);
    for i = 1:length(fields)
        field = fields{i};
        if isnumeric(network.(field)) && ~isscalar(network.(field))
            network.(field) = gpuArray(network.(field));
        end
    end
end
