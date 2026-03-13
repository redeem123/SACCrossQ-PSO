function metrics = updateGlobalPlanningMetrics(metrics, currentTime, globalPlanDuration, replanReason, globalPath, algorithmSpecificStats)
    % Update metrics after global planning - optimized with pre-allocation checking
    
    % Initialize or extend pre-allocated arrays
    if ~isfield(metrics, 'globalPlanTimes') || isempty(metrics.globalPlanTimes)
        % Pre-allocate for typical number of replanning events (estimated 50)
        metrics.globalPlanTimes = zeros(50, 1);
        metrics.globalPlanDurations = zeros(50, 1);
        metrics.pathLengths = zeros(50, 1);
        metrics.pathWaypoints = zeros(50, 1);
        metrics.reasons = cell(50, 1);
        metrics.currentIndex = 1;
    end
    
    idx = metrics.currentIndex;
    
    % Extend arrays if needed
    if idx > length(metrics.globalPlanTimes)
        newSize = length(metrics.globalPlanTimes) * 2;
        metrics.globalPlanTimes(newSize) = 0;
        metrics.globalPlanDurations(newSize) = 0;
        metrics.pathLengths(newSize) = 0;
        metrics.pathWaypoints(newSize) = 0;
        metrics.reasons{newSize} = '';
    end
    
    % Update metrics at current index
    metrics.globalPlanTimes(idx) = currentTime;
    metrics.globalPlanDurations(idx) = globalPlanDuration;
    metrics.reasons{idx} = replanReason;
    
    % Use actual PSO fitness - this ensures consistency with displayed values
    globalPathFitness = algorithmSpecificStats.actualBestFitness;

    metrics.pathLengths(idx) = globalPathFitness;
    metrics.pathWaypoints(idx) = size(globalPath, 1);

    % Increment index for next update
    metrics.currentIndex = idx + 1;

    % Store actualBestFitness at top level for CSV compatibility (RL algorithms)
    if isfield(algorithmSpecificStats, 'actualBestFitness')
        metrics.actualBestFitness = algorithmSpecificStats.actualBestFitness;
    end

    % Store fitness components if available
    if isfield(algorithmSpecificStats, 'fitnessComponents')
        metrics.fitnessComponents = algorithmSpecificStats.fitnessComponents;
    end
    
    % Store convergence history if available
    if ~isempty(algorithmSpecificStats) && isfield(algorithmSpecificStats, 'convergenceHistory')
        metrics.convergenceHistory = algorithmSpecificStats.convergenceHistory;
    end
    
    % Store parameter history if available
    if ~isempty(algorithmSpecificStats) && isfield(algorithmSpecificStats, 'parameterHistory')
        if isfield(algorithmSpecificStats.parameterHistory, 'w')
            metrics.w_history = algorithmSpecificStats.parameterHistory.w;
        end
        if isfield(algorithmSpecificStats.parameterHistory, 'c1')
            metrics.c1_history = algorithmSpecificStats.parameterHistory.c1;
        end
        if isfield(algorithmSpecificStats.parameterHistory, 'c2')
            metrics.c2_history = algorithmSpecificStats.parameterHistory.c2;
        end
        if isfield(algorithmSpecificStats.parameterHistory, 'w_samples')
            metrics.w_samples = algorithmSpecificStats.parameterHistory.w_samples;
        end
        if isfield(algorithmSpecificStats.parameterHistory, 'c1_samples')
            metrics.c1_samples = algorithmSpecificStats.parameterHistory.c1_samples;
        end
        if isfield(algorithmSpecificStats.parameterHistory, 'c2_samples')
            metrics.c2_samples = algorithmSpecificStats.parameterHistory.c2_samples;
        end
    end
    
    % Update algorithm-specific metrics only if they exist
    if ~isempty(algorithmSpecificStats)
        switch metrics.algorithmName
           
                
            case 'SVPSO'
                if isfield(algorithmSpecificStats, 'sphericalStats')
                    metrics.sphericalConvergence = [metrics.sphericalConvergence; algorithmSpecificStats.sphericalStats];
                end
                if isfield(algorithmSpecificStats, 'vectorStats')
                    metrics.vectorMagnitudes = [metrics.vectorMagnitudes; algorithmSpecificStats.vectorStats];
                end
                
            
        end
    end
end

