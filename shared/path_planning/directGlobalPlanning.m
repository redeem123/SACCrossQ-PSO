function [finalPath, pathLength, executionTime, timeHistory, metrics] = directGlobalPlanning(startPoint, goalPoint, dangerZones, terrainGrid, terrainX, terrainY, mapSize, varargin)
    % Direct Global Planning - Pure PSO optimization without hierarchical planning
    %
    % Simplified planning function that runs PSO algorithms once without
    % dynamic replanning, UAV simulation, or time-based iterations.
    %
    % This is ideal for:
    %   - Pure algorithm comparison
    %   - Static optimization problems
    %   - Benchmarking PSO variants
    %
    % Parameters:
    %   The first parameter in varargin should be the algorithm name
    %   Followed by algorithm-specific parameters

    tic; % Start timer

    % Extract algorithm name from varargin
    if isempty(varargin)
        error('Algorithm name must be specified as first parameter in varargin');
    end

    algorithmName = varargin{1};
    if iscell(algorithmName)
        algorithmName = algorithmName{1};
    end

    % Remove algorithm name from varargin for further processing
    remainingParams = varargin(2:end);

    % Parse algorithm-specific parameters
    [algorithmParams, commonParams] = parseAlgorithmParameters(algorithmName, remainingParams);

    % Initialize metrics structure
    metrics = struct();
    metrics.algorithmName = algorithmName;

    % Single global planning call (no replanning, no simulation)
    fprintf('\n========== Direct Global Planning ==========\n');
    fprintf('Algorithm: %s\n', algorithmName);
    fprintf('Start: [%.1f, %.1f, %.1f]\n', startPoint);
    fprintf('Goal:  [%.1f, %.1f, %.1f]\n', goalPoint);
    fprintf('============================================\n\n');

    globalPlanStartTime = tic;

    % Call PSO algorithm once
    [globalPath, algorithmSpecificStats] = callGlobalPlanningAlgorithm( ...
        algorithmName, startPoint, goalPoint, dangerZones, ...
        terrainGrid, terrainX, terrainY, mapSize, algorithmParams);

    % Record planning duration
    globalPlanDuration = toc(globalPlanStartTime);

    % Store results in metrics structure
    metrics.globalPlanDurations = globalPlanDuration;
    metrics.globalPlanTimes = 0;  % Single planning at time 0
    metrics.reasons = {'Initial planning'};

    % Calculate path length
    pathLength = calculatePathLength(globalPath);
    metrics.pathLengths = pathLength;
    metrics.pathWaypoints = size(globalPath, 1);

    % Store algorithm-specific results
    if isfield(algorithmSpecificStats, 'actualBestFitness')
        metrics.actualBestFitness = algorithmSpecificStats.actualBestFitness;
    end

    if isfield(algorithmSpecificStats, 'fitnessComponents')
        metrics.fitnessComponents = algorithmSpecificStats.fitnessComponents;
    end

    if isfield(algorithmSpecificStats, 'convergenceHistory')
        metrics.convergenceHistory = algorithmSpecificStats.convergenceHistory;
    end

    % Store parameter history if available (for RL-based PSO)
    if isfield(algorithmSpecificStats, 'parameterHistory')
        if isfield(algorithmSpecificStats.parameterHistory, 'w')
            metrics.w_history = algorithmSpecificStats.parameterHistory.w;
        end
        if isfield(algorithmSpecificStats.parameterHistory, 'c1')
            metrics.c1_history = algorithmSpecificStats.parameterHistory.c1;
        end
        if isfield(algorithmSpecificStats.parameterHistory, 'c2')
            metrics.c2_history = algorithmSpecificStats.parameterHistory.c2;
        end
    end

    % Use global path as final path (no UAV simulation)
    finalPath = globalPath;

    % Store final global path in metrics (required by compare_algorithms)
    metrics.finalGlobalPath = globalPath;

    % No time history (not a simulation)
    timeHistory = 0;

    % Total execution time
    executionTime = toc;

    % Display final results
    fprintf('\n========== Planning Complete ==========\n');
    fprintf('Algorithm: %s\n', algorithmName);
    fprintf('Execution time: %.4f seconds\n', executionTime);
    fprintf('Path length: %.4f units\n', pathLength);
    fprintf('Waypoints: %d\n', size(globalPath, 1));

    if isfield(metrics, 'actualBestFitness')
        fprintf('Best fitness: %.4f\n', metrics.actualBestFitness);
    end

    fprintf('========================================\n\n');
end
