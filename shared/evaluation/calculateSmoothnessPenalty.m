function smoothness = calculateSmoothnessPenalty(path)
    % Calculate smoothness penalty based on angular changes
    smoothness = 0;
    
    if size(path, 1) < 3
        return;
    end
    
    for i = 2:size(path, 1)-1
        v1 = path(i,:) - path(i-1,:);
        v2 = path(i+1,:) - path(i,:);
        
        if norm(v1) > 0 && norm(v2) > 0
            v1_norm = v1 / norm(v1);
            v2_norm = v2 / norm(v2);
            
            % Calculate angle between vectors
            angle = acos(max(-1, min(1, dot(v1_norm, v2_norm))));
            
            % Penalty increases with sharper turns
            smoothness = smoothness + (pi - angle) * 10;
        end
    end
end

