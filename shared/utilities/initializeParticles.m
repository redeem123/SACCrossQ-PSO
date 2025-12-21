function particles = initializeParticles(popSize, dims, numWaypoints, mapSize, ...
    terrainGrid, terrainX, terrainY, startPoint, goalPoint, obstacles, trees)
    % Initialize particle swarm
    
    particles = struct();
    particles.cartesianPositions = zeros(popSize, dims);
    particles.velocities = zeros(popSize, dims);
    particles.bestPositions = zeros(popSize, dims);
    particles.bestFitness = Inf(popSize, 1);
    particles.fitness = Inf(popSize, 1);
    particles.lastImprovement = ones(popSize, 1);
    
    for i = 1:popSize
        for j = 1:numWaypoints
            x = rand() * mapSize(1);
            y = rand() * mapSize(2);
            
            terrainHeight = getTerrainHeight(x, y, terrainGrid, terrainX, terrainY);
            
            z = terrainHeight + 8 + rand() * 20;
            waypoint = [x, y, z];
            waypoint = max([1, 1, 1], min(waypoint, mapSize));
            
            idx = (j-1)*3 + 1;
            particles.cartesianPositions(i, idx:idx+2) = waypoint;
        end
        
        particles.velocities(i,:) = (rand(1, dims) - 0.5) * 3;
        [particles.fitness(i), ~] = evaluatePathFitness(particles.cartesianPositions(i,:), ...
            startPoint, goalPoint, obstacles, trees, terrainGrid, terrainX, terrainY, numWaypoints);
        
        particles.bestPositions(i,:) = particles.cartesianPositions(i,:);
        particles.bestFitness(i) = particles.fitness(i);
    end
end

