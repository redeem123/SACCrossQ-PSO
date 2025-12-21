function safetyCost = calculateSafetyCost(path, obstacles, trees, terrainGrid, terrainX, terrainY)
    % Calculate Safety Cost (C_safety)
    safetyCost = struct();
    
    % Obstacle Clearance
    safetyCost.obstacleDistance = calculateMinObstacleDistance(path, obstacles);
    
    % Terrain Clearance
    safetyCost.terrainClearance = calculateMinTerrainClearance(path, terrainGrid, terrainX, terrainY);
    
    % Collision Risk
    safetyCost.collisionRisk = calculateCollisionRisk(path, obstacles, trees, terrainGrid, terrainX, terrainY);
    
    % Emergency Maneuver Space
    safetyCost.emergencySpace = calculateEmergencyManeuverSpace(path, obstacles, trees);
    
    % Total Safety Cost (lower is better)
    safetyCost.total = (1.0 / max(safetyCost.obstacleDistance, 0.1)) + ...
                       (1.0 / max(safetyCost.terrainClearance, 0.1)) + ...
                       safetyCost.collisionRisk + ...
                       (1.0 / max(safetyCost.emergencySpace, 0.1));
end

