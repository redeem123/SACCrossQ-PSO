function bestRunIdx = findBestRun(allRunResults, algorithms)
    % Find the run with the best overall performance for visualization
    numRuns = length(allRunResults);
    runScores = zeros(numRuns, 1);
    
    for run = 1:numRuns
        totalScore = 0;
        validAlgs = 0;
        
        for alg = 1:length(algorithms)
            algName = algorithms{alg}.fieldName;
            if isfield(allRunResults{run}, algName)
                pathLength = allRunResults{run}.(algName).pathLength;
                totalScore = totalScore + pathLength;
                validAlgs = validAlgs + 1;
            end
        end
        
        if validAlgs > 0
            runScores(run) = totalScore / validAlgs;  % Average path length
        else
            runScores(run) = inf;
        end
    end
    
    [~, bestRunIdx] = min(runScores);
end


