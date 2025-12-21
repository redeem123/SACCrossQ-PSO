function [projectedPoint, distance] = projectPointOnSegment(p, v1, v2)
    % Project point p onto line segment v1-v2 and return projected point and distance
    
    % Explicitly check and reshape inputs to ensure compatibility
    % First, ensure all are row vectors with 3 components
    if numel(p) >= 3
        p = reshape(p(1:3), 1, 3);
    else
        % Pad with zeros if necessary
        p_temp = zeros(1, 3);
        p_temp(1:numel(p)) = p(:)';
        p = p_temp;
    end
    
    if numel(v1) >= 3
        v1 = reshape(v1(1:3), 1, 3);
    else
        v1_temp = zeros(1, 3);
        v1_temp(1:numel(v1)) = v1(:)';
        v1 = v1_temp;
    end
    
    if numel(v2) >= 3
        v2 = reshape(v2(1:3), 1, 3);
    else
        v2_temp = zeros(1, 3);
        v2_temp(1:numel(v2)) = v2(:)';
        v2 = v2_temp;
    end
    
    % Now all vectors are 1x3 and operations should work
    % Vector from v1 to v2
    v1v2 = v2 - v1;
    
    % Vector from v1 to p
    v1p = p - v1;
    
    % Check if projection falls on segment
    if norm(v1v2) < 1e-10  % Avoid division by zero if points are the same
        projectedPoint = v1;
        distance = norm(p - v1);
        return;
    end
    
    % Calculate parameter t
    t = sum(v1p .* v1v2) / sum(v1v2 .* v1v2);
    
    if t < 0
        % p is closest to v1
        projectedPoint = v1;
    elseif t > 1
        % p is closest to v2
        projectedPoint = v2;
    else
        % p is closest to point on segment
        projectedPoint = v1 + t * v1v2;
    end
    
    distance = norm(p - projectedPoint);
end

