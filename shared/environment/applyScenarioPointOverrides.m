function environment = applyScenarioPointOverrides(environment, scenarioDefinition)
    % Apply optional start/goal point overrides from a scenario definition.

    if nargin < 2 || isempty(scenarioDefinition)
        return;
    end

    if isfield(scenarioDefinition, 'startPoint') && ~isempty(scenarioDefinition.startPoint)
        environment.startPoint = double(reshape(scenarioDefinition.startPoint, 1, 3));
    end

    if isfield(scenarioDefinition, 'goalPoint') && ~isempty(scenarioDefinition.goalPoint)
        environment.goalPoint = double(reshape(scenarioDefinition.goalPoint, 1, 3));
    end
end
