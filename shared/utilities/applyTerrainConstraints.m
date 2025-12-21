function constrainedPosition = applyTerrainConstraints(cartesianPosition, terrainGrid, terrainX, terrainY, mapSize, numWaypoints)
    % Apply terrain and boundary constraints to cartesian waypoint positions
    % Used in spherical coordinate-based algorithms (SVPSO, SAEPSO)

    constrainedPosition = cartesianPosition;

    for j = 1:numWaypoints
        idx = (j-1)*3 + 1;
        waypoint = constrainedPosition(idx:idx+2);

        % Apply map boundary constraints
        waypoint = max([1, 1, 1], min(waypoint, mapSize));

        % Apply terrain clearance constraint
        terrainHeight = getTerrainHeight(waypoint(1), waypoint(2), terrainGrid, terrainX, terrainY);
        waypoint(3) = max(waypoint(3), terrainHeight + 10); % Minimum 10 units above terrain

        constrainedPosition(idx:idx+2) = waypoint;
    end
end