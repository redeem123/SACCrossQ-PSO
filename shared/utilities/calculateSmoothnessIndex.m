function smoothnessIndex = calculateSmoothnessIndex(path)
    % Calculate smoothness index (higher = smoother, 0-1 scale)
    if size(path, 1) < 3
        smoothnessIndex = 1.0; % Perfect smoothness for short paths
        return;
    end
    
    totalAngularChange = 0;
    segmentCount = 0;
    
    for i = 2:size(path, 1)-1
        v1 = path(i,:) - path(i-1,:);
        v2 = path(i+1,:) - path(i,:);
        
        if norm(v1) > 0 && norm(v2) > 0
            v1_norm = v1 / norm(v1);
            v2_norm = v2 / norm(v2);
            
            % Calculate angle between vectors
            angle = acos(max(-1, min(1, dot(v1_norm, v2_norm))));
            
            % Accumulate angular changes
            totalAngularChange = totalAngularChange + abs(pi - angle);
            segmentCount = segmentCount + 1;
        end
    end
    
    if segmentCount > 0
        avgAngularChange = totalAngularChange / segmentCount;
        % Convert to smoothness index (0 = very rough, 1 = very smooth)
        smoothnessIndex = builtin('exp', -avgAngularChange); % Exponential decay
    else
        smoothnessIndex = 1.0;
    end
end

