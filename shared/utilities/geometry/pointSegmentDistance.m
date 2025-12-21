function distance = pointSegmentDistance(p, v1, v2)
    % Calculate minimum distance from point p to line segment defined by v1 and v2
    
    % Vector from v1 to v2
    v1v2 = v2 - v1;
    
    % Vector from v1 to p
    v1p = p - v1;
    
    % Check if projection falls on segment
    if norm(v1v2) < 1e-10  % Avoid division by zero if points are the same
        distance = norm(p - v1);
        return;
    end
    
    t = dot(v1p, v1v2) / dot(v1v2, v1v2);
    
    if t < 0
        % p is closest to v1
        distance = norm(p - v1);
    elseif t > 1
        % p is closest to v2
        distance = norm(p - v2);
    else
        % p is closest to point on segment
        projection = v1 + t * v1v2;
        distance = norm(p - projection);
    end
end

