function scenario = generateCurriculumScenario(levelDef, mapSize)
    % Generate training scenario based on curriculum difficulty level
    % EXACTLY MATCHES run_comparison.m environment setup
    % Only varies the NUMBER of trees/obstacles based on curriculum level
    %
    % Inputs:
    %   levelDef: Level definition structure from CurriculumManager
    %   mapSize: Map dimensions [x, y, z] (default: [400, 400, 100])
    %
    % Outputs:
    %   scenario: Complete scenario structure with all environment elements

    if nargin < 2
        mapSize = [400, 400, 100];
    end

    fprintf('[generateCurriculumScenario] START - Level: %s\n', levelDef.name);

    scenario = struct();
    scenario.mapSize = mapSize;
    scenario.levelName = levelDef.name;

    % === GENERATE TERRAIN (EXACT SAME AS run_comparison.m) ===
    [terrainGrid, terrainX, terrainY] = generateFixedTerrain(mapSize, 1);

    % === GENERATE START AND GOAL (SCALED FROM run_comparison.m) ===
    % run_comparison.m uses: startPoint = [10, 95, 10], goalPoint = [97, 2, 10]
    % for mapSize = [100, 100, 100]
    % Scale to current mapSize (usually [400, 400, 100])
    scaleFactor = mapSize(1) / 100;

    startPoint = [10 * scaleFactor, 95 * scaleFactor, 10];  % Bottom-left, altitude 10
    goalPoint = [97 * scaleFactor, 2 * scaleFactor, 10];    % Top-right, altitude 10

    scenario.startPoint = startPoint;
    scenario.goalPoint = goalPoint;
    scenario.actualDistance = norm(goalPoint - startPoint);

    % === DETERMINE COUNTS BASED ON CURRICULUM LEVEL ===
    % Curriculum only varies the NUMBER of obstacles/trees
    % Mimics run_comparison.m scenarios: 0, 30, 60, 90
    % NOTE: generateFixedObstacles maxes out at 15 obstacles
    switch levelDef.name
        case 'Easy'
            numTrees = randi([0, 10]);      % 0-10 trees
            numObstacles = randi([0, 3]);   % 0-3 obstacles

        case 'Medium'
            numTrees = randi([20, 35]);     % 20-35 trees
            numObstacles = randi([5, 8]);   % 5-8 obstacles

        case 'Hard'
            numTrees = randi([50, 70]);     % 50-70 trees
            numObstacles = randi([10, 12]); % 10-12 obstacles

        case 'Expert'
            numTrees = randi([80, 100]);    % 80-100 trees
            numObstacles = randi([13, 15]); % 13-15 obstacles (max)

        otherwise
            numTrees = 30;
            numObstacles = 8;
    end

    % === GENERATE TREES (EXACT SAME AS run_comparison.m) ===
    trees = generateFixedTrees(numTrees, terrainGrid, terrainX, terrainY, mapSize);

    fprintf('[DEBUG generateCurriculumScenario] trees size after generateFixedTrees: %d x %d\n', size(trees, 1), size(trees, 2));

    scenario.trees = trees;
    scenario.numTrees = numTrees;

    % === GENERATE OBSTACLES (EXACT SAME AS run_comparison.m) ===
    obstacles = generateFixedObstacles(numObstacles, mapSize, terrainGrid, terrainX, terrainY);

    scenario.obstacles = obstacles;
    scenario.numObstacles = numObstacles;

    % === STORE TERRAIN ===
    scenario.terrainGrid = terrainGrid;
    scenario.terrainX = terrainX;
    scenario.terrainY = terrainY;

    % === CALCULATE DIFFICULTY SCORE ===
    scenario.difficulty = assessScenarioDifficulty(scenario);

    % === METADATA ===
    scenario.generatedAt = datetime('now');
    scenario.levelDef = levelDef;
end
