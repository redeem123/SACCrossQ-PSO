function writeSummaryStatisticsCSV(allRunResults, algorithms, filename)
    % Write summary statistics to CSV with complete fitness breakdown including duplicate penalty
    fid = fopen(filename, 'w');
    if fid == -1
        error('Could not open CSV file for writing: %s', filename);
    end
    
    % Header
    fprintf(fid, 'Algorithm,SuccessRate,Mean_GlobalFitness,Std_GlobalFitness,Median_GlobalFitness,Min_GlobalFitness,Max_GlobalFitness,Mean_PathLength,Std_PathLength,Mean_TurningPenalty,Std_TurningPenalty,Mean_ClimbingPenalty,Std_ClimbingPenalty,Mean_HeightPenalty,Std_HeightPenalty,Mean_CollisionPenalty,Std_CollisionPenalty,Mean_TerrainPenalty,Std_TerrainPenalty,Mean_DangerZonePenalty,Std_DangerZonePenalty,Mean_DuplicatePenalty,Std_DuplicatePenalty,Mean_TotalGlobalPlanTime,Std_TotalGlobalPlanTime,Mean_AvgGlobalPlanTime,Mean_NumGlobalPlans\n');
    
    % Calculate and write statistics for each algorithm
    for alg = 1:length(algorithms)
        algName = algorithms{alg}.fieldName;
        displayName = algorithms{alg}.displayName;
        
        globalFitness = [];
        pathLengths = [];
        turningPenalties = [];
        climbingPenalties = [];
        heightPenalties = [];
        collisionPenalties = [];
        terrainPenalties = [];
        dangerZonePenalties = [];  % New component
        duplicatePenalties = [];  % New component
        totalGlobalPlanTimes = [];
        avgGlobalPlanTimes = [];
        numGlobalPlans = [];
        
        for run = 1:length(allRunResults)
            if isfield(allRunResults{run}, algName)
                % Get fitness components if available
                if isfield(allRunResults{run}.(algName), 'fitnessComponents')
                    components = allRunResults{run}.(algName).fitnessComponents;
                    globalFitness = [globalFitness; components.totalFitness];
                    pathLengths = [pathLengths; components.pathLength];
                    turningPenalties = [turningPenalties; components.turningPenalty];
                    climbingPenalties = [climbingPenalties; components.climbingPenalty];
                    heightPenalties = [heightPenalties; components.heightPenalty];
                    collisionPenalties = [collisionPenalties; components.collisionPenalty];
                    terrainPenalties = [terrainPenalties; components.terrainPenalty];
                    
                    if isfield(components, 'dangerZonePenalty')
                        dangerZonePenalties = [dangerZonePenalties; components.dangerZonePenalty];
                    else
                        dangerZonePenalties = [dangerZonePenalties; 0];
                    end

                    if isfield(components, 'duplicatePenalty')
                        duplicatePenalties = [duplicatePenalties; components.duplicatePenalty];
                    else
                        duplicatePenalties = [duplicatePenalties; 0];
                    end
                elseif isfield(allRunResults{run}.(algName), 'actualBestFitness')
                    % For RL algorithms (RRSACPSO, RLAM-PSO) that use actualBestFitness
                    globalFitness = [globalFitness; allRunResults{run}.(algName).actualBestFitness];
                    pathLengths = [pathLengths; allRunResults{run}.(algName).actualBestFitness];
                    turningPenalties = [turningPenalties; 0];
                    climbingPenalties = [climbingPenalties; 0];
                    heightPenalties = [heightPenalties; 0];
                    collisionPenalties = [collisionPenalties; 0];
                    terrainPenalties = [terrainPenalties; 0];
                    dangerZonePenalties = [dangerZonePenalties; 0];
                    duplicatePenalties = [duplicatePenalties; 0];
                elseif isfield(allRunResults{run}.(algName), 'pathLengths')
                    % Use pathLength as fallback for globalFitness
                    pathLengthsData = allRunResults{run}.(algName).pathLengths;
                    if ~isempty(pathLengthsData)
                        globalFitness = [globalFitness; pathLengthsData(end)];
                    else
                        globalFitness = [globalFitness; NaN];
                    end
                    pathLengths = [pathLengths; 0];
                    turningPenalties = [turningPenalties; 0];
                    climbingPenalties = [climbingPenalties; 0];
                    heightPenalties = [heightPenalties; 0];
                    collisionPenalties = [collisionPenalties; 0];
                    terrainPenalties = [terrainPenalties; 0];
                    dangerZonePenalties = [dangerZonePenalties; 0];
                    duplicatePenalties = [duplicatePenalties; 0];  % New component
                end
                
                % Get global planning time information
                if isfield(allRunResults{run}.(algName), 'globalPlanDurations')
                    globalPlanDurations = allRunResults{run}.(algName).globalPlanDurations;
                    totalGlobalPlanTimes = [totalGlobalPlanTimes; sum(globalPlanDurations)];
                    avgGlobalPlanTimes = [avgGlobalPlanTimes; mean(globalPlanDurations)];
                    numGlobalPlans = [numGlobalPlans; length(globalPlanDurations)];
                else
                    totalGlobalPlanTimes = [totalGlobalPlanTimes; 0];
                    avgGlobalPlanTimes = [avgGlobalPlanTimes; 0];
                    numGlobalPlans = [numGlobalPlans; 0];
                end
            end
        end
        
        if ~isempty(globalFitness)
            % Calculate success rate (percentage of runs with finite fitness)
            successfulRuns = isfinite(globalFitness);
            successRate = sum(successfulRuns) / length(globalFitness);
            
            % Filter data to include only successful runs for statistics
            validGlobalFitness = globalFitness(successfulRuns);
            validPathLengths = pathLengths(successfulRuns);
            validTurningPenalties = turningPenalties(successfulRuns);
            validClimbingPenalties = climbingPenalties(successfulRuns);
            validHeightPenalties = heightPenalties(successfulRuns);
            validCollisionPenalties = collisionPenalties(successfulRuns);
            validTerrainPenalties = terrainPenalties(successfulRuns);
            validDangerZonePenalties = dangerZonePenalties(successfulRuns);
            validDuplicatePenalties = duplicatePenalties(successfulRuns);
            validTotalGlobalPlanTimes = totalGlobalPlanTimes(successfulRuns);
            validAvgGlobalPlanTimes = avgGlobalPlanTimes(successfulRuns);
            validNumGlobalPlans = numGlobalPlans(successfulRuns);
            
            if isempty(validGlobalFitness)
                % Handle case where all runs failed
                fprintf(fid, '%s,%.4f,Inf,NaN,Inf,Inf,Inf,Inf,NaN,Inf,NaN,Inf,NaN,Inf,NaN,Inf,NaN,Inf,NaN,Inf,NaN,Inf,NaN,Inf,NaN,Inf,0.00\n', ...
                    displayName, successRate);
            else
                fprintf(fid, '%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f\n', ...
                    displayName, successRate, ...
                    mean(validGlobalFitness), std(validGlobalFitness), median(validGlobalFitness), ...
                    min(validGlobalFitness), max(validGlobalFitness), ...
                    mean(validPathLengths), std(validPathLengths), ...
                    mean(validTurningPenalties), std(validTurningPenalties), ...
                    mean(validClimbingPenalties), std(validClimbingPenalties), ...
                    mean(validHeightPenalties), std(validHeightPenalties), ...
                    mean(validCollisionPenalties), std(validCollisionPenalties), ...
                    mean(validTerrainPenalties), std(validTerrainPenalties), ...
                    mean(validDangerZonePenalties), std(validDangerZonePenalties), ...
                    mean(validDuplicatePenalties), std(validDuplicatePenalties), ...
                    mean(validTotalGlobalPlanTimes), std(validTotalGlobalPlanTimes), mean(validAvgGlobalPlanTimes), mean(validNumGlobalPlans));
            end
        end
    end
    
    fclose(fid);
    fprintf('Summary statistics written to: %s\n', filename);
end

