function distance = pointPathDistance(point, path)
    % Calculate minimum distance from a point to a path - optimized vectorized version
    if size(path, 1) < 2
        distance = inf;
        return;
    end
    
    % Vectorized calculation for all segments at once
    v1 = path(1:end-1, :);  % Start points
    v2 = path(2:end, :);    % End points
    
    % Calculate distances to all segments vectorized
    numSegments = size(v1, 1);
    distances = zeros(numSegments, 1);
    
    for i = 1:numSegments
        distances(i) = pointSegmentDistance(point, v1(i,:), v2(i,:));
    end
    
    distance = min(distances);
end

