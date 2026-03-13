function path = constructPathFromPSO(position, startPoint, goalPoint, numWaypoints, mapSize)
    % Construct a path from PSO particle position
    if nargin < 5
        mapSize = [];
    end

    % Extract waypoints from position vector
    waypoints = zeros(numWaypoints, 3);
    for i = 1:numWaypoints
        idx = (i-1)*3 + 1;
        waypoint = position(idx:idx+2);
        if ~isempty(mapSize)
            waypoint(1) = max(1, min(waypoint(1), mapSize(1)));
            waypoint(2) = max(1, min(waypoint(2), mapSize(2)));
            waypoint(3) = max(1, min(waypoint(3), mapSize(3)));
        end
        waypoints(i,:) = waypoint;
    end

    if ~isempty(mapSize)
        startPoint(1) = max(1, min(startPoint(1), mapSize(1)));
        startPoint(2) = max(1, min(startPoint(2), mapSize(2)));
        startPoint(3) = max(1, min(startPoint(3), mapSize(3)));
        goalPoint(1) = max(1, min(goalPoint(1), mapSize(1)));
        goalPoint(2) = max(1, min(goalPoint(2), mapSize(2)));
        goalPoint(3) = max(1, min(goalPoint(3), mapSize(3)));
    end

    % Create full path with start and goal
    path = [startPoint; waypoints; goalPoint];
    % disp(path);  % Disabled to reduce console clutter during training
end

%% Cost Analysis Functions
