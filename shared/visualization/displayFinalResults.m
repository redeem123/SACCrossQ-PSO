function displayFinalResults(algorithmName, executionTime, pathLength, currentPosition, goalPoint, metrics)
    % Display final results (common format for all algorithms)
    disp(['Total execution time: ', num2str(executionTime), ' seconds']);
    disp(['Path length: ', num2str(pathLength), ' units']);
    disp(['Path Quality Cost: ', num2str(metrics.costAnalysis.pathCost.total)]);
    disp(['Safety Cost: ', num2str(metrics.costAnalysis.safetyCost.total)]);
    disp(['Computational Cost: ', num2str(metrics.costAnalysis.computationalCost.total)]);
    
    % Algorithm-specific result display
    switch algorithmName
        case 'DynamicPSO'
            if ~isempty(metrics.w_history)
                disp('Dynamic PSO Parameter Statistics:');
                disp(['  Inertia weight (w) range: ', num2str(min(metrics.w_history)), ' to ', num2str(max(metrics.w_history))]);
                disp(['  Cognitive coefficient (c1) range: ', num2str(min(metrics.c1_history)), ' to ', num2str(max(metrics.c1_history))]);
                disp(['  Social coefficient (c2) range: ', num2str(min(metrics.c2_history)), ' to ', num2str(max(metrics.c2_history))]);
            end
            
        case 'SVPSO'
            if ~isempty(metrics.sphericalConvergence)
                disp(['Average spherical convergence: ', num2str(mean(metrics.sphericalConvergence))]);
                disp(['Average vector magnitude: ', num2str(mean(metrics.vectorMagnitudes))]);
            end
    end
    
    % Final status
    if norm(currentPosition - goalPoint) > 2.0
        disp('Warning: UAV did not reach the goal within the allotted time.');
    else
        disp('Success: UAV reached the goal!');
    end
    
    % Planning statistics
    disp(['Number of global replanning operations: ', num2str(length(metrics.globalPlanTimes))]);
    disp(['Average global planning duration: ', num2str(mean(metrics.globalPlanDurations)), ' seconds']);
end

