function results = runAblationConfiguration(config, startPoint, goalPoint, obstacles, ...
    trees, obstacleDynamics, terrainGrid, terrainX, terrainY, mapSize, numRuns)
    % Run multiple trials for a single ablation configuration
    
    results = struct();
    results.fitness = zeros(numRuns, 1);
    results.pathLength = zeros(numRuns, 1);
    results.executionTime = zeros(numRuns, 1);
    results.convergenceHistory = cell(numRuns, 1);
    results.neuralInfluence = zeros(numRuns, 1);
    
    % Common parameters
    popSize = 20;
    maxIterations = 80;
    fixedW = 0.7;
    fixedC1 = 1.5;
    fixedC2 = 1.5;
    
    % Pre-allocate temporary arrays for parfor
    fitness_temp = zeros(numRuns, 1);
    pathLength_temp = zeros(numRuns, 1);
    executionTime_temp = zeros(numRuns, 1);
    convergenceHistory_temp = cell(numRuns, 1);
    neuralInfluence_temp = zeros(numRuns, 1);
    
    parfor run = 1:numRuns
        % Set random seed for reproducibility
        rng(run * 1000);
        
        % Run ablated version of Neural-Guided PSO
        [globalPath, convergenceHistory, algorithmSpecificStats] = ...
            globalPathPlanningNeuralGuidedPSO_Ablated(startPoint, goalPoint, ...
            obstacles, trees, terrainGrid, terrainX, terrainY, mapSize, ...
            popSize, maxIterations, fixedW, fixedC1, fixedC2, config);
        
        % Store results in temporary variables
        fitness_temp(run) = algorithmSpecificStats.actualBestFitness;
        pathLength_temp(run) = calculatePathLength(globalPath);
        executionTime_temp(run) = algorithmSpecificStats.executionTime;
        convergenceHistory_temp{run} = convergenceHistory;
        
        if isfield(algorithmSpecificStats, 'avgNeuralInfluence')
            neuralInfluence_temp(run) = algorithmSpecificStats.avgNeuralInfluence;
        end
    end
    
    % Assign temporary variables to results structure
    results.fitness = fitness_temp;
    results.pathLength = pathLength_temp;
    results.executionTime = executionTime_temp;
    results.convergenceHistory = convergenceHistory_temp;
    results.neuralInfluence = neuralInfluence_temp;
    
    % Calculate statistics
    results.meanFitness = mean(results.fitness);
    results.stdFitness = std(results.fitness);
    results.meanPathLength = mean(results.pathLength);
    results.stdPathLength = std(results.pathLength);
    results.meanExecutionTime = mean(results.executionTime);
    results.meanNeuralInfluence = mean(results.neuralInfluence);
end

