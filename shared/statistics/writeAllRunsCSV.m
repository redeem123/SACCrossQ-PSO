function writeAllRunsCSV(allRunResults, algorithms, filename)
    % Fixed version of writeAllRunsCSV that handles missing fitness components
    fid = fopen(filename, 'w');
    if fid == -1
        error('Could not open CSV file for writing: %s', filename);
    end
    
    % Header with all fitness components
    fprintf(fid, 'Run,Algorithm,Success,GlobalPathFitness,PathLength,TurningPenalty,ClimbingPenalty,HeightPenalty,CollisionPenalty,TerrainPenalty,DangerZonePenalty,DuplicatePenalty,TotalGlobalPlanTime,AvgGlobalPlanTime,NumGlobalPlans\n');
    
    % Data
    for run = 1:length(allRunResults)
        for alg = 1:length(algorithms)
            algName = algorithms{alg}.fieldName;
            displayName = algorithms{alg}.displayName;
            
            if isfield(allRunResults{run}, algName)
                % Initialize default values
                globalPathFitness = 0;
                pathLength = 0; 
                turningPenalty = 0;
                climbingPenalty = 0;
                heightPenalty = 0;
                collisionPenalty = 0;
                terrainPenalty = 0;
                dangerZonePenalty = 0;
                duplicatePenalty = 0;
                
                % Get fitness components if available
                if isfield(allRunResults{run}.(algName), 'fitnessComponents') && ~isempty(allRunResults{run}.(algName).fitnessComponents)
                    components = allRunResults{run}.(algName).fitnessComponents;
                    globalPathFitness = getFieldValue(components, 'totalFitness', 0);
                    pathLength = getFieldValue(components, 'pathLength', 0);
                    turningPenalty = getFieldValue(components, 'turningPenalty', 0);
                    climbingPenalty = getFieldValue(components, 'climbingPenalty', 0);
                    heightPenalty = getFieldValue(components, 'heightPenalty', 0);
                    collisionPenalty = getFieldValue(components, 'collisionPenalty', 0);
                    terrainPenalty = getFieldValue(components, 'terrainPenalty', 0);
                    dangerZonePenalty = getFieldValue(components, 'dangerZonePenalty', 0);
                    duplicatePenalty = getFieldValue(components, 'duplicatePenalty', 0);
                elseif isfield(allRunResults{run}.(algName), 'actualBestFitness')
                    % For RL algorithms (AFSACPSO, RLAM-PSO) that use actualBestFitness
                    globalPathFitness = allRunResults{run}.(algName).actualBestFitness;
                    pathLength = globalPathFitness;
                elseif isfield(allRunResults{run}.(algName), 'pathLength')
                    % Fallback to pathLength if components not available
                    globalPathFitness = allRunResults{run}.(algName).pathLength;
                    pathLength = globalPathFitness;
                else
                    % Skip this entry if no data available
                    continue;
                end
                
                % Get global planning time information
                totalGlobalPlanTime = 0;
                avgGlobalPlanTime = 0;
                numGlobalPlans = 0;
                
                % Determine success (finite fitness)
                success = isfinite(globalPathFitness);

                if isfield(allRunResults{run}.(algName), 'globalPlanDurations') && ~isempty(allRunResults{run}.(algName).globalPlanDurations)
                    globalPlanDurations = allRunResults{run}.(algName).globalPlanDurations;
                    totalGlobalPlanTime = sum(globalPlanDurations);
                    avgGlobalPlanTime = mean(globalPlanDurations);
                    numGlobalPlans = length(globalPlanDurations);
                end
                
                fprintf(fid, '%d,%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n', ...
                    run, displayName, success, globalPathFitness, pathLength, turningPenalty, climbingPenalty, heightPenalty, ...
                    collisionPenalty, terrainPenalty, dangerZonePenalty, duplicatePenalty, totalGlobalPlanTime, avgGlobalPlanTime, numGlobalPlans);
            end
        end
    end
    
    fclose(fid);
    fprintf('All runs data written to: %s\n', filename);
end

