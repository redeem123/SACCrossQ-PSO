function cartesianPos = sphericalToCartesian(sphericalPos, startPoint, numWaypoints)
    % Correct implementation of Equation 16 from the paper
    cartesianPos = zeros(1, numWaypoints * 3);
    currentPos = startPoint;
    
    for i = 1:numWaypoints
        idx = (i-1)*3 + 1;
        rho = sphericalPos(idx);     % Distance
        xi = sphericalPos(idx+1);    % Climbing angle  
        phi = sphericalPos(idx+2);   % Turning angle
        
        % Equation 16: use PREVIOUS position to calculate current
        deltaX = rho * cos(xi) * cos(phi);
        deltaY = rho * cos(xi) * sin(phi);
        deltaZ = rho * sin(xi);
        
        newPos = currentPos + [deltaX, deltaY, deltaZ];
        cartesianPos(idx:idx+2) = newPos;
        currentPos = newPos; % Update for next iteration
    end
end
%% 8. Dynamic PSO with CNN

