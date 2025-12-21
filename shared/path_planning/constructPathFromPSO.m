function path = constructPathFromPSO(position, startPoint, goalPoint, numWaypoints, mapSize)
    % Construct a path from PSO particle position
    % All waypoints are constrained to stay within mapSize boundaries

    % Handle optional mapSize parameter
    if nargin < 5
        mapSize = [];  % No constraint if not provided
    end

    % Extract waypoints from position vector
    waypoints = zeros(numWaypoints, 3);
    for i = 1:numWaypoints
        idx = (i-1)*3 + 1;
        waypoint = position(idx:idx+2);

        % Apply boundary constraints if mapSize provided
        if ~isempty(mapSize)
            % Clamp X, Y, Z to be within [1, mapSize]
            waypoint(1) = max(1, min(waypoint(1), mapSize(1)));
            waypoint(2) = max(1, min(waypoint(2), mapSize(2)));
            waypoint(3) = max(1, min(waypoint(3), mapSize(3)));
        end

        waypoints(i,:) = waypoint;
    end

    % Ensure start and goal are within bounds too
    startPoint = constrainPointToMapSize(startPoint, mapSize);
    goalPoint = constrainPointToMapSize(goalPoint, mapSize);

    % Create full path with start and goal
    path = [startPoint; waypoints; goalPoint];
end

function point = constrainPointToMapSize(point, mapSize)
    % Constrain a single point to mapSize boundaries
    if nargin < 2 || isempty(mapSize)
        return;  % No constraint if mapSize not provided
    end

    point(1) = max(1, min(point(1), mapSize(1)));
    point(2) = max(1, min(point(2), mapSize(2)));
    point(3) = max(1, min(point(3), mapSize(3)));
end

%% Cost Analysis Functions
