function [startPoint, goalPoint, obstacles, trees, terrainGrid, terrainX, terrainY, mapSize] = generateTrainingScenario(config, episode)
    % Generate training scenario MATCHING run_comparison.m exactly
    % Uses same functions as compare_algorithms.m uses
    %
    % Inputs:
    %   config: Configuration from RLAMPSO_Config
    %   episode: Current episode number (for curriculum progression)
    %
    % Outputs:
    %   startPoint: [x, y, z] starting position
    %   goalPoint: [x, y, z] goal position
    %   obstacles: [N x 4] matrix [x, y, z, radius] for dynamic obstacles
    %   trees: [M x 7] matrix [x, y, z, trunkRadius, canopyRadius, isCylinder, height]
    %   terrainGrid: Height map for terrain
    %   terrainX, terrainY: Grid coordinates
    %   mapSize: [width, height, maxAltitude]

    % Get map size from config
    mapSize = config.mapSize;

    % === GENERATE TERRAIN (EXACT SAME AS run_comparison.m) ===
    [terrainGrid, terrainX, terrainY] = generateFixedTerrain(mapSize, 1);

    % === GENERATE START AND GOAL (SCALED FROM run_comparison.m) ===
    % run_comparison.m uses: startPoint = [10, 95, 10], goalPoint = [97, 2, 10]
    % for mapSize = [100, 100, 100]
    % Scale to current mapSize (usually [400, 400, 100])
    scaleFactor = mapSize(1) / 100;

    startPoint = [10 * scaleFactor, 95 * scaleFactor, 10];  % Bottom-left, altitude 10
    goalPoint = [97 * scaleFactor, 2 * scaleFactor, 10];    % Top-right, altitude 10

    % === VARY OBSTACLE/TREE COUNTS BASED ON EPISODE ===
    % Increase difficulty over training episodes (curriculum learning)
    episodeFraction = episode / 500;  % Assuming ~500 episodes
    episodeFraction = min(episodeFraction, 1.0);

    % Curriculum progression: start easy, gradually increase difficulty
    numTrees = round(10 + episodeFraction * 80);      % 10 → 90 trees
    numObstacles = round(5 + episodeFraction * 10);   % 5 → 15 obstacles
    % === GENERATE TREES (EXACT SAME AS run_comparison.m) ===
    trees = generateFixedTrees(numTrees, terrainGrid, terrainX, terrainY, mapSize);

    % === GENERATE OBSTACLES (EXACT SAME AS run_comparison.m) ===
    obstacles = generateFixedObstacles(numObstacles, mapSize, terrainGrid, terrainX, terrainY);
end
