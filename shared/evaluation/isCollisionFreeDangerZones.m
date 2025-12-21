function isFree = isCollisionFreeDangerZones(point1, point2, dangerZones, terrainGrid, terrainX, terrainY)
    % Check if path segment between point1 and point2 is collision-free
    % Danger zones are infinite-height cylinders from ground to infinity

    % Sample points along the segment
    numSamples = 10;
    for i = 0:numSamples
        alpha = i / numSamples;
        point = point1 + alpha * (point2 - point1);

        % Check terrain collision
        terrainHeight = getTerrainHeight(point(1), point(2), terrainGrid, terrainX, terrainY);
        if point(3) < terrainHeight + 8  % Minimum clearance of 8 units
            isFree = false;
            return;
        end

        % Check danger zone collision (infinite-height cylinders)
        for j = 1:size(dangerZones, 1)
            zoneX = dangerZones(j,1);
            zoneY = dangerZones(j,2);
            zoneRadius = dangerZones(j,3);

            % Calculate horizontal distance to danger zone center
            horizontalDist = norm([point(1) - zoneX, point(2) - zoneY]);

            % Check if point is inside danger zone cylinder
            if horizontalDist < zoneRadius
                isFree = false;
                return;
            end
        end
    end

    isFree = true;
end
