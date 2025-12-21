function dynamics = initializeFixedObstacleDynamics(numObstacles, mapSize)
    % Initialize fixed dynamics parameters for obstacles
    dynamics = struct();
    
    % Fixed velocity patterns
    velocityPatterns = [
        0.5, 0.3, 0.1;
        -0.3, 0.4, 0.2;
        0.2, -0.5, 0.3;
        -0.4, -0.2, 0.4;
        0.3, 0.2, -0.5;
        -0.2, 0.5, -0.3;
        0.4, -0.4, 0.2;
        -0.5, 0.1, 0.4;
        0.1, -0.3, -0.4;
        0.0, 0.0, 0.0;  % stationary
        -0.3, -0.3, 0.5;
        0.5, 0.0, -0.2;
        -0.2, 0.4, 0.0;
        0.3, -0.5, -0.1;
        0.0, 0.0, 0.0   % stationary
    ];
    
    oscillationPatterns = [
        2.0, 1.0, 0.5;
        1.5, 2.0, 1.0;
        1.0, 0.5, 2.0;
        0.5, 1.0, 1.5;
        2.0, 2.0, 0.0;
        0.0, 1.5, 1.5;
        1.5, 0.0, 2.0;
        2.0, 1.5, 0.0;
        0.5, 2.0, 1.0;
        0.0, 0.0, 0.0;  % stationary
        1.0, 1.0, 1.0;
        2.0, 0.5, 0.0;
        0.0, 2.0, 0.5;
        1.5, 0.0, 1.0;
        0.0, 0.0, 0.0   % stationary
    ];
    
    frequencyPatterns = [
        0.05, 0.10, 0.15;
        0.10, 0.15, 0.05;
        0.15, 0.05, 0.10;
        0.07, 0.12, 0.03;
        0.12, 0.03, 0.07;
        0.03, 0.07, 0.12;
        0.08, 0.04, 0.14;
        0.04, 0.14, 0.08;
        0.14, 0.08, 0.04;
        0.00, 0.00, 0.00;  % stationary
        0.06, 0.11, 0.02;
        0.11, 0.02, 0.06;
        0.02, 0.06, 0.11;
        0.09, 0.13, 0.07;
        0.00, 0.00, 0.00   % stationary
    ];
    
    phasePatterns = [
        0.0, 1.0, 2.0;
        2.0, 0.0, 1.0;
        1.0, 2.0, 0.0;
        0.5, 1.5, 2.5;
        2.5, 0.5, 1.5;
        1.5, 2.5, 0.5;
        0.0, 3.0, 1.5;
        3.0, 1.5, 0.0;
        1.5, 0.0, 3.0;
        0.0, 0.0, 0.0;  % stationary
        0.7, 1.8, 2.9;
        1.8, 2.9, 0.7;
        2.9, 0.7, 1.8;
        0.3, 2.1, 1.2;
        0.0, 0.0, 0.0   % stationary
    ];
    
    % Scale velocities based on map size
    maxSpeed = min(mapSize) * 0.05; % Maximum 5% of map size per second
    
    % Use patterns for the specified number of obstacles
    numPatterns = size(velocityPatterns, 1);
    
    dynamics.velocities = zeros(numObstacles, 3);
    dynamics.oscillations = zeros(numObstacles, 3);
    dynamics.frequencies = zeros(numObstacles, 3);
    dynamics.phases = zeros(numObstacles, 3);
    
    for i = 1:numObstacles
        patternIdx = mod(i-1, numPatterns) + 1; % Cycle through patterns if numObstacles > numPatterns
        
        dynamics.velocities(i,:) = velocityPatterns(patternIdx,:) * maxSpeed;
        dynamics.oscillations(i,:) = oscillationPatterns(patternIdx,:);
        dynamics.frequencies(i,:) = frequencyPatterns(patternIdx,:);
        dynamics.phases(i,:) = phasePatterns(patternIdx,:);
    end
end






%% Helper Functions for PSO Algorithms
