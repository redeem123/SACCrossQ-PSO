function particles = updateParticlePosition(particles, i, mapSize, terrainGrid, ...
    terrainX, terrainY, numWaypoints)
    % Update particle position with velocity clamping and constraints
    
    maxVelocity = 0.15 * (mapSize(1) + mapSize(2) + mapSize(3)) / 3;
    
    for j = 1:numWaypoints
        idx = (j-1)*3 + 1;
        segmentVel = particles.velocities(i, idx:idx+2);
        velMag = norm(segmentVel);
        
        if velMag > maxVelocity
            particles.velocities(i, idx:idx+2) = segmentVel * (maxVelocity / velMag);
        end
    end
    
    particles.cartesianPositions(i,:) = particles.cartesianPositions(i,:) + particles.velocities(i,:);
    
    for j = 1:numWaypoints
        idx = (j-1)*3 + 1;
        waypoint = particles.cartesianPositions(i, idx:idx+2);
        waypoint = max([1, 1, 1], min(waypoint, mapSize));
        
        terrainHeight = getTerrainHeight(waypoint(1), waypoint(2), terrainGrid, terrainX, terrainY);
        waypoint(3) = max(waypoint(3), terrainHeight + 10);
        
        particles.cartesianPositions(i, idx:idx+2) = waypoint;
    end
end

