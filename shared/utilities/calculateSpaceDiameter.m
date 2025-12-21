function spaceDiameter = calculateSpaceDiameter(positions)
    % Calculate approximate space diameter for normalization
    if size(positions, 1) < 2
        spaceDiameter = 100; % Default value
        return;
    end
    
    minPos = min(positions, [], 1);
    maxPos = max(positions, [], 1);
    spaceDiameter = norm(maxPos - minPos);
    
    if spaceDiameter < 1e-6
        spaceDiameter = 100; % Default if particles are clustered
    end
end

