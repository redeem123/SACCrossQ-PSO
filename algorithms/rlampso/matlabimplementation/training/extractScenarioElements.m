function [startPoint, goalPoint, obstacles, trees, terrainGrid, terrainX, terrainY] = extractScenarioElements(scenario)
    startPoint = scenario.startPoint;
    goalPoint = scenario.goalPoint;
    obstacles = scenario.obstacles;
    trees = scenario.trees;
    terrainGrid = scenario.terrainGrid;
    terrainX = scenario.terrainX;
    terrainY = scenario.terrainY;
end
