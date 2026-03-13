function action = predictAction(actorNet, state, useGPU)
    % Predict action using DL Toolbox actor network
    %
    % Replaces forwardPassPaperActor.m with modern dlnetwork prediction
    % Handles automatic GPU placement and batch processing
    %
    % Inputs:
    %   actorNet: dlnetwork actor network
    %   state: State vector (stateSize × 1) or batch (stateSize × batchSize)
    %   useGPU: Use GPU if available (default: auto-detect)
    %
    % Outputs:
    %   action: Action vector (actionSize × 1) or batch (actionSize × batchSize)
    %           Values in range [-1, 1] due to tanh output
    %
    % Usage:
    %   action = predictAction(actorNet, state);              % Single state
    %   actions = predictAction(actorNet, stateBatch, true);  % Batch with GPU

    if nargin < 3
        useGPU = canUseGPU();  % Auto-detect GPU
    end

    % Convert to dlarray with appropriate format
    if size(state, 2) == 1
        % Single state: 'CB' format (Channel-Batch)
        dlState = dlarray(state, 'CB');
    else
        % Batch of states: 'CB' format (Channel-Batch)
        dlState = dlarray(state, 'CB');
    end

    % Move to GPU if requested
    if useGPU && canUseGPU()
        dlState = gpuArray(dlState);
    end

    % Forward pass through actor network
    dlAction = predict(actorNet, dlState);

    % Convert back to regular array
    if isgpuarray(dlAction)
        action = gather(extractdata(dlAction));
    else
        action = extractdata(dlAction);
    end
end
