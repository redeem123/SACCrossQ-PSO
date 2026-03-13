function runDynamicPSOAblationStudy()
    % Main function for ablation study of NNPSO (Neural-Guided PSO)

    % Add paths to categorized folders
    % Get current directory and navigate to project root
    currentDir = fileparts(mfilename('fullpath'));
    % Go up from ablation -> neural_guided_pso -> algorithms -> project root
    projectRoot = fileparts(fileparts(fileparts(currentDir)));

    addpath(genpath(fullfile(projectRoot, 'algorithms')));
    addpath(genpath(fullfile(projectRoot, 'shared', 'utilities')));
    addpath(genpath(fullfile(projectRoot, 'shared', 'environment')));

    % Create output folder
    resultsDir = fullfile(projectRoot, 'results');
    if ~exist(resultsDir, 'dir')
        mkdir(resultsDir);
    end

    fprintf('Starting Ablation Study for Neural-Guided PSO...\n\n');

    % Initialize parallel pool
    initializeParallelPool();
    
    % Get ablation configurations
    ablationConfigs = defineCompleteAblationConfigurations();
    
    % Define test scenarios
    scenarios = [
        0, 0;
        30, 30;    % Medium complexity
        60, 60;    % High complexity
        90, 90;
    ];
    
    % Study parameters
    numRuns = 10;  % Number of runs per configuration
    
    % Pre-allocate results storage
    ablationResults = struct();
    
    % Run ablation study for each scenario
    for scenarioIdx = 1:size(scenarios, 1)
        numTrees = scenarios(scenarioIdx, 1);
        numObstacles = scenarios(scenarioIdx, 2);
        
        fprintf('\n========== SCENARIO %d: %d Trees, %d Obstacles ==========\n', ...
                scenarioIdx, numTrees, numObstacles);
        
        % Environment setup
        environment = resolveTerrainEnvironment(struct( ...
            'mapSize', [100, 100, 100], ...
            'startPoint', [0, 100, 10], ...
            'goalPoint', [100, 0, 10], ...
            'terrainFile', ''));
        mapSize = environment.mapSize;
        startPoint = environment.startPoint;
        goalPoint = environment.goalPoint;
        
        % Generate environment
        [terrainGrid, terrainX, terrainY] = generateFixedTerrain( ...
            mapSize, scenarioIdx, environment.terrainFile);
        trees = generateFixedTrees(numTrees, terrainGrid, terrainX, terrainY, mapSize);
        obstacles = generateFixedObstacles(numObstacles, mapSize, terrainGrid, terrainX, terrainY);
        obstacleDynamics = initializeFixedObstacleDynamics(numObstacles, mapSize);
        
        % Run each ablation configuration
        scenarioResults = struct();
        
        for configIdx = 1:length(ablationConfigs)
            config = ablationConfigs{configIdx};
            fprintf('\n--- Testing Configuration: %s ---\n', config.name);
            
            configResults = runAblationConfiguration(config, startPoint, goalPoint, ...
                obstacles, trees, obstacleDynamics, terrainGrid, terrainX, terrainY, ...
                mapSize, numRuns);
            
            scenarioResults.(matlab.lang.makeValidName(config.name)) = configResults;
        end
        
        ablationResults.(['Scenario' num2str(scenarioIdx)]) = scenarioResults;
    end
    
    % Analyze and visualize results
    analyzeAblationResults(ablationResults, ablationConfigs, scenarios);
    generateAblationReport(ablationResults, ablationConfigs, scenarios);
end

