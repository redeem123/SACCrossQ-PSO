function dangerZones = generateDangerZones(numZones, mapSize, terrainGrid, terrainX, terrainY)
    % Generate danger zones as infinite-height cylinders from ground to infinity
    % Danger zones: [x, y, radius] - UAV cannot fly through these vertical cylinders
    %#ok<INUSD>

    dangerZones = zeros(numZones, 3);  % [x, y, radius]

    % Define fixed danger zone positions and radii
    dangerZonePositions = [
        30, 30, 5.0;   % 1
        50, 50, 4.0;   % 2
        70, 30, 6.0;   % 3
        40, 70, 4.5;   % 4
        25, 60, 3.5;   % 5
        65, 65, 5.5;   % 6
        85, 45, 4.0;   % 7
        55, 85, 4.5;   % 8
        45, 25, 4.2;   % 9
        75, 75, 5.0;   % 10
        20, 80, 4.0;   % 11
        80, 20, 4.5;   % 12
        60, 40, 3.5;   % 13
        40, 60, 4.0;   % 14
        90, 90, 5.0    % 15
    ];
    xyScale = [mapSize(1) / 100, mapSize(2) / 100];
    radiusScale = mean(xyScale);
    dangerZonePositions(:, 1) = dangerZonePositions(:, 1) * xyScale(1);
    dangerZonePositions(:, 2) = dangerZonePositions(:, 2) * xyScale(2);
    dangerZonePositions(:, 3) = dangerZonePositions(:, 3) * radiusScale;

    numPredefined = size(dangerZonePositions, 1);

    % Use fixed positions for available predefined zones
    for i = 1:min(numZones, numPredefined)
        dangerZones(i,:) = dangerZonePositions(i, :);
    end
    
    % Randomly generate any additional zones required
    if numZones > numPredefined
        marginX = max(10, 0.1 * mapSize(1));
        marginY = max(10, 0.1 * mapSize(2));
        for i = (numPredefined + 1):numZones
            % Generate random position within a padded interior to avoid edges
            x = marginX + rand() * max(mapSize(1) - 2 * marginX, eps);
            y = marginY + rand() * max(mapSize(2) - 2 * marginY, eps);
            radius = (3.0 + rand() * 3.0) * radiusScale;  % Random radius scaled to map size
            
            dangerZones(i,:) = [x, y, radius];
        end
    end
end
