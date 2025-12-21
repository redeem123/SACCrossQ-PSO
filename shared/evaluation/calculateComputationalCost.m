function compCost = calculateComputationalCost(planningTimes, localPlanningTimes, memoryUsage)
    % Calculate Computational Cost (C_computation)
    compCost = struct();
    
    % Planning Time
    compCost.totalPlanningTime = sum(planningTimes);
    compCost.avgPlanningTime = mean(planningTimes);
    compCost.maxPlanningTime = max(planningTimes);
    
    % No local planning anymore
    compCost.totalLocalTime = 0;
    compCost.avgLocalTime = 0;
    
    % Memory Usage (estimated)
    compCost.memoryUsage = memoryUsage;
    
    % Total Computational Cost
    compCost.total = compCost.totalPlanningTime + compCost.memoryUsage;
end

