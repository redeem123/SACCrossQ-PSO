function [difficultyScore, metrics] = assessScenarioDifficulty(scenario)
    % Automatically assess scenario difficulty based on multiple factors
    %
    % Difficulty Metrics:
    %   - Obstacle density (weight: 0.4)
    %   - Tree density (weight: 0.3)
    %   - Terrain variance (weight: 0.2)
    %   - Normalized distance (weight: 0.1)
    %
    % Inputs:
    %   scenario: Scenario structure with obstacles, trees, terrain, etc.
    %
    % Outputs:
    %   difficultyScore: Weighted difficulty score [0, 1]
    %   metrics: Detailed breakdown of difficulty components

    metrics = struct();

    % Map dimensions
    mapSize = scenario.mapSize;
    mapArea = mapSize(1) * mapSize(2);

    % === OBSTACLE DENSITY ===
    numObstacles = scenario.numObstacles;

    % Calculate total obstacle area
    obstacleArea = 0;
    if ~isempty(scenario.obstacles)
        for i = 1:size(scenario.obstacles, 1)
            radius = scenario.obstacles(i, 4);
            obstacleArea = obstacleArea + pi * radius^2;
        end
    end

    obstacleDensity = obstacleArea / mapArea;
    metrics.obstacleDensity = obstacleDensity;

    % Normalize to [0, 1] (assume max 10% coverage is very high)
    obstacleDensityNorm = min(1.0, obstacleDensity / 0.10);

    % === TREE DENSITY ===
    numTrees = scenario.numTrees;

    % Calculate total tree canopy area
    treeArea = 0;
    if ~isempty(scenario.trees)
        for i = 1:size(scenario.trees, 1)
            canopyRadius = scenario.trees(i, 5);
            treeArea = treeArea + pi * canopyRadius^2;
        end
    end

    treeDensity = treeArea / mapArea;
    metrics.treeDensity = treeDensity;

    % Normalize to [0, 1] (assume max 5% coverage is very high)
    treeDensityNorm = min(1.0, treeDensity / 0.05);

    % === TERRAIN VARIANCE ===
    terrainGrid = scenario.terrainGrid;

    % Calculate terrain statistics
    terrainMean = mean(terrainGrid(:));
    terrainStd = std(terrainGrid(:));
    terrainRange = max(terrainGrid(:)) - min(terrainGrid(:));

    % Calculate terrain roughness (gradient magnitude)
    [Gx, Gy] = gradient(terrainGrid);
    terrainRoughness = mean(sqrt(Gx(:).^2 + Gy(:).^2));

    metrics.terrainStd = terrainStd;
    metrics.terrainRange = terrainRange;
    metrics.terrainRoughness = terrainRoughness;

    % Normalize terrain variance (assume std > 15 is very high)
    terrainVarianceNorm = min(1.0, terrainStd / 15.0);

    % === DISTANCE DIFFICULTY ===
    distance = scenario.actualDistance;

    % Normalize distance (assume 500 is maximum challenging distance)
    distanceNorm = min(1.0, distance / 500.0);
    metrics.distance = distance;
    metrics.distanceNorm = distanceNorm;

    % === PATH COMPLEXITY ===
    % Estimate path complexity by checking obstacle interference

    pathComplexity = 0;
    if ~isempty(scenario.obstacles)
        % Direct line from start to goal
        startPoint = scenario.startPoint;
        goalPoint = scenario.goalPoint;

        % Count obstacles within corridor
        corridorWidth = 50;  % Width of direct path corridor

        for i = 1:size(scenario.obstacles, 1)
            obsPos = scenario.obstacles(i, 1:2);
            obsRadius = scenario.obstacles(i, 4);

            % Distance from obstacle to line segment
            dist = pointToLineDistance(obsPos, startPoint(1:2), goalPoint(1:2));

            if dist < (corridorWidth + obsRadius)
                pathComplexity = pathComplexity + 1;
            end
        end

        % Normalize by number of obstacles
        if numObstacles > 0
            pathComplexity = pathComplexity / numObstacles;
        end
    end

    metrics.pathComplexity = pathComplexity;

    % === COMBINED DIFFICULTY SCORE ===
    % Weighted combination of factors
    weights = struct();
    weights.obstacle = 0.4;
    weights.tree = 0.3;
    weights.terrain = 0.2;
    weights.distance = 0.1;

    difficultyScore = weights.obstacle * obstacleDensityNorm + ...
                      weights.tree * treeDensityNorm + ...
                      weights.terrain * terrainVarianceNorm + ...
                      weights.distance * distanceNorm;

    % Ensure score is in [0, 1]
    difficultyScore = max(0, min(1, difficultyScore));

    % === CATEGORICAL DIFFICULTY ===
    if difficultyScore < 0.25
        difficultyCategory = 'Easy';
    elseif difficultyScore < 0.50
        difficultyCategory = 'Medium';
    elseif difficultyScore < 0.75
        difficultyCategory = 'Hard';
    else
        difficultyCategory = 'Expert';
    end

    metrics.difficultyScore = difficultyScore;
    metrics.difficultyCategory = difficultyCategory;
    metrics.weights = weights;

    % Store individual components
    metrics.components = struct();
    metrics.components.obstacleDensity = obstacleDensityNorm;
    metrics.components.treeDensity = treeDensityNorm;
    metrics.components.terrainVariance = terrainVarianceNorm;
    metrics.components.distance = distanceNorm;
end

function dist = pointToLineDistance(point, lineStart, lineEnd)
    % Calculate minimum distance from point to line segment
    %
    % Inputs:
    %   point: [x, y] point coordinates
    %   lineStart: [x, y] line start
    %   lineEnd: [x, y] line end
    %
    % Output:
    %   dist: Minimum distance

    % Vector from start to end
    lineVec = lineEnd - lineStart;
    lineLen = norm(lineVec);

    if lineLen < 1e-6
        % Degenerate line segment
        dist = norm(point - lineStart);
        return;
    end

    % Normalized line direction
    lineDir = lineVec / lineLen;

    % Vector from start to point
    pointVec = point - lineStart;

    % Project point onto line
    projection = dot(pointVec, lineDir);

    if projection <= 0
        % Closest point is line start
        dist = norm(point - lineStart);
    elseif projection >= lineLen
        % Closest point is line end
        dist = norm(point - lineEnd);
    else
        % Closest point is on the line segment
        closestPoint = lineStart + projection * lineDir;
        dist = norm(point - closestPoint);
    end
end
