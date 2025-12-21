function costAnalysis = calculateComprehensiveCostAnalysis(finalPath, obstacles, trees, terrainGrid, terrainX, terrainY, metrics)
    % Calculate comprehensive cost analysis (common to all algorithms)
    disp('Calculating comprehensive cost analysis...');
    
    % Path Quality Cost
    costAnalysis.pathCost = calculatePathQualityCost(finalPath, terrainGrid, terrainX, terrainY);
    
    % Safety Cost
    costAnalysis.safetyCost = calculateSafetyCost(finalPath, obstacles, trees, terrainGrid, terrainX, terrainY);
    
    % Computational Cost
    estimatedMemoryUsage = (length(metrics.globalPlanTimes) * 50 + size(finalPath,1) * 10) / 1024; % MB
    costAnalysis.computationalCost = calculateComputationalCost(metrics.globalPlanDurations, [], estimatedMemoryUsage);
end

