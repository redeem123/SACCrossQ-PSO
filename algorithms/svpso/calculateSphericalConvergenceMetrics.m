function convergence = calculateSphericalConvergenceMetrics(sphericalPositions, globalBestSpherical)
    % Calculate convergence measure specific to spherical coordinates
    
    % Calculate distances in spherical space (considering the different nature of each coordinate)
    numParticles = size(sphericalPositions, 1);
    sphericalDims = size(sphericalPositions, 2);
    
    totalSphericalDistance = 0;
    
    for i = 1:numParticles
        particleDistance = 0;
        
        % Process each waypoint's spherical coordinates separately
        for waypoint = 1:sphericalDims/3
            idx = (waypoint-1)*3 + 1;
            
            % Extract (Ï?, Î¾, Ï†) for this waypoint
            rho_p = sphericalPositions(i, idx);
            xi_p = sphericalPositions(i, idx+1);
            phi_p = sphericalPositions(i, idx+2);
            
            rho_g = globalBestSpherical(idx);
            xi_g = globalBestSpherical(idx+1);
            phi_g = globalBestSpherical(idx+2);
            
            % Calculate spherical distance considering coordinate meanings
            % Ï? difference (linear)
            rho_diff = abs(rho_p - rho_g);
            
            % Angular differences (considering periodicity for Ï†)
            xi_diff = abs(xi_p - xi_g);
            phi_diff = min(abs(phi_p - phi_g), 2*pi - abs(phi_p - phi_g));
            
            % Weighted spherical distance for this waypoint
            waypointDistance = sqrt(rho_diff^2 + (rho_p * xi_diff)^2 + (rho_p * cos(xi_p) * phi_diff)^2);
            particleDistance = particleDistance + waypointDistance;
        end
        
        totalSphericalDistance = totalSphericalDistance + particleDistance;
    end
    
    % Normalize by number of particles and waypoints
    avgSphericalDistance = totalSphericalDistance / (numParticles * (sphericalDims/3));
    
    % Convert to convergence measure (higher = more converged)
    maxPossibleDistance = 100; % Approximate maximum spherical distance
    convergence = 1 - min(avgSphericalDistance / maxPossibleDistance, 1);
end


%% 5. PSO Simple Adaptation

