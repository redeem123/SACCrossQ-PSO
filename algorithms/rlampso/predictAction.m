function action = predictAction(actorNet, state, ~)
    % Predict action using DL Toolbox actor network (CPU-only)
    %
    % Inputs:
    %   actorNet: dlnetwork actor
    %   state:    stateSize × 1 (or stateSize × batchSize)
    %
    % Outputs:
    %   action:   actionSize × 1 (or actionSize × batchSize), values in [-1, 1]

    dlState = dlarray(state, 'CB');
    dlAction = predict(actorNet, dlState);
    action = extractdata(dlAction);
end
