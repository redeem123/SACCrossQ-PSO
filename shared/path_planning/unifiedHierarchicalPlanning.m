function [finalPath, pathLength, executionTime, timeHistory, metrics] = unifiedHierarchicalPlanning(startPoint, goalPoint, initialObstacles, trees, obstacleDynamics, terrainGrid, terrainX, terrainY, mapSize, varargin)
    % Unified hierarchical planning function that supports all PSO variants
    % 
    % Parameters:
    %   The first parameter in varargin should be the algorithm name
    %   Followed by algorithm-specific parameters and common parameters
    
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
    
    % Extract common parameters
    globalPlanInterval = commonParams.globalPlanInterval;
    pathDeviationThreshold = commonParams.pathDeviationThreshold;
    obstacleChangeThreshold = commonParams.obstacleChangeThreshold;
    timeStep = commonParams.timeStep;
    totalTime = commonParams.totalTime;
    terrain_map = commonParams.terrain_map;
    
    % Initialize variables for dynamic simulation
    currentTime = 0;
    lastGlobalPlanTime = -globalPlanInterval; % Force initial global planning
    currentPosition = startPoint;
    finalPath = currentPosition;
    obstacles = initialObstacles;
    prevObstacles = initialObstacles;
    timeHistory = currentTime;
    
    % Initialize unified metrics structure
    metrics = initializeUnifiedMetrics(algorithmName);
    
    % UAV dynamics parameters
    uavSpeed = 5; % units per second
    
    % Initialize global path
    globalPath = [startPoint; goalPoint]; % Direct path initially
    
    % Main unified simulation loop
    while currentTime < totalTime && norm(currentPosition - goalPoint) > 2.0
        
        % Update obstacle positions based on dynamics
        obstacles = updateObstaclePositions(initialObstacles, obstacleDynamics, currentTime, mapSize, terrainGrid, terrainX, terrainY);
        
        % Decide if global replanning is needed
        [needGlobalReplan, replanReason] = checkReplanningConditions(currentTime, lastGlobalPlanTime, currentPosition, globalPath, obstacles, prevObstacles, globalPlanInterval, pathDeviationThreshold, obstacleChangeThreshold);

        % Perform global planning if needed
        if needGlobalReplan
            disp(['Executing ', algorithmName, ' global path planning. Reason: ', replanReason]);
            globalPlanStartTime = tic;
            
            % Call appropriate global planning algorithm with convergence tracking
            [globalPath, algorithmSpecificStats] = callGlobalPlanningAlgorithm(algorithmName, currentPosition, goalPoint, obstacles, trees, terrainGrid, terrainX, terrainY, mapSize, algorithmParams);
            
            % Record global planning metrics
            globalPlanDuration = toc(globalPlanStartTime);
            metrics = updateGlobalPlanningMetrics(metrics, currentTime, globalPlanDuration, replanReason, globalPath, algorithmSpecificStats);
            
            lastGlobalPlanTime = currentTime;
            disp(['Global planning completed in ', num2str(globalPlanDuration), ' seconds']);
            disp(['Path length: ', num2str(calculatePathLength(globalPath)), ' units with ', num2str(size(globalPath, 1)), ' waypoints']);
        end
        
        % Always perform local path refinement (common to all algorithms)
        % Directly follow global path without local planning
        nextPosition = followGlobalPath(currentPosition, globalPath, uavSpeed, timeStep);
        
        % Update metrics and position
        [metrics, currentPosition, finalPath] = updateSimulationState(metrics, currentTime, 0, nextPosition, currentPosition, finalPath, obstacles, terrainGrid, terrainX, terrainY);
        
        % Store current obstacles as previous for next iteration
        prevObstacles = obstacles;
        
        % Update time
        currentTime = currentTime + timeStep;
        timeHistory = [timeHistory; currentTime];

        % Visualize current state (disabled for performance in comparison mode)
        % Uncomment below lines to enable real-time visualization:
        % visualizeCurrentStateTerrain(finalPath, globalPath, obstacles, trees, terrainGrid, terrainX, terrainY, startPoint, goalPoint, mapSize, currentTime, needGlobalReplan, terrain_map);
        % drawnow;
        % pause(0.01);
    end
    
    % Calculate final results
    pathLength = calculatePathLength(finalPath);
    executionTime = toc;
    
    % Calculate comprehensive cost analysis (common to all algorithms)
    metrics.costAnalysis = calculateComprehensiveCostAnalysis(finalPath, obstacles, trees, terrainGrid, terrainX, terrainY, metrics);
    
    % Display final results
    displayFinalResults(algorithmName, executionTime, pathLength, currentPosition, goalPoint, metrics);
    
    % Store final global path
    metrics.finalGlobalPath = globalPath;
end

%% Path Planning Utility Functions
