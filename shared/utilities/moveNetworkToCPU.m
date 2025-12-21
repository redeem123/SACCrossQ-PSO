function network = moveNetworkToCPU(network)
    % Move all network weights and biases from GPU to CPU
    %
    % Inputs:
    %   network: Struct containing network weights (W1, b1, W2, b2, etc.)
    %
    % Returns:
    %   network: Same struct with weights moved to CPU

    fields = fieldnames(network);
    for i = 1:length(fields)
        field = fields{i};
        if isa(network.(field), 'gpuArray')
            network.(field) = gather(network.(field));
        end
    end
end
