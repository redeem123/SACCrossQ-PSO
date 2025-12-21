function metrics = initializeUnifiedMetrics(algorithmName)
    % Initialize metrics structure for any algorithm
    metrics = struct();
    metrics.globalPlanTimes = [];
    metrics.globalPlanDurations = [];
    metrics.reasons = {};
    metrics.pathLengths = [];
    metrics.pathWaypoints = [];
    metrics.obstacleDistances = [];
    metrics.terrainClearances = [];
    metrics.algorithmName = algorithmName;
    
    % Algorithm-specific metrics - initialize all possible fields
    switch algorithmName

        case 'SVPSO'
            metrics.sphericalConvergence = [];
            metrics.vectorMagnitudes = [];
    end
    
    % Initialize cost analysis structure
    metrics.costAnalysis = struct();
    metrics.costAnalysis.pathCost = struct();
    metrics.costAnalysis.safetyCost = struct();
    metrics.costAnalysis.computationalCost = struct();
end

