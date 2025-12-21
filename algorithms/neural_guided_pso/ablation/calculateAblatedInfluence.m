function gamma = calculateAblatedInfluence(eliteNetwork, eliteParticles, particles, ...
    particleIdx, iter, maxIterations, config)
    % Calculate neural network influence based on configuration
    
    if ~config.adaptiveInfluence
        % Fixed influence
        gamma = config.maxGamma;
    else
        % Adaptive influence (original implementation)
        if size(eliteParticles, 1) > 10 && ~isempty(eliteNetwork.lossHistory)
            networkReadiness = min(1.0, size(eliteParticles, 1) / 50);
            recentLoss = mean(eliteNetwork.lossHistory(max(1,end-9):end));
            
            if isfinite(recentLoss) && recentLoss < 10
                networkConfidence = exp(-recentLoss);
                gamma = config.maxGamma * networkConfidence * networkReadiness;
            else
                gamma = 0;
            end
            
            % Apply particle ranking if enabled
            if ~isfield(config, 'useParticleRanking') || config.useParticleRanking
                [~, sortedIndices] = sort(particles.fitness);
                particleRank = find(sortedIndices == particleIdx);
                popSize = length(particles.fitness);
                
                if particleRank <= popSize * 0.2
                    gamma = gamma * 0.5;  % Top performers: less guidance
                elseif particleRank >= popSize * 0.8
                    gamma = min(gamma * 1.5, config.maxGamma);  % Bottom: more guidance
                end
            end
        else
            gamma = 0;
        end
    end
    
    gamma = max(0, min(config.maxGamma, gamma));
end

