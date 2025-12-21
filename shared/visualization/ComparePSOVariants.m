function ComparePSOVariants(results, algorithms)
    % Check if all algorithms are present
    missingAlgorithms = {};
    for i = 1:length(algorithms)
        if ~isfield(results, algorithms{i}.fieldName)  % Using cell array indexing
            missingAlgorithms{end+1} = algorithms{i}.displayName;  % Using cell array indexing
        end
    end
    
    if ~isempty(missingAlgorithms)
        disp('Cannot compare PSO variants: one or more algorithms missing');
        disp('Missing algorithms:');
        for i = 1:length(missingAlgorithms)
            disp(['  - ' missingAlgorithms{i}]);
        end
        disp('Available algorithms:');
        fields = fieldnames(results);
        for i = 1:length(fields)
            disp(['  - ' fields{i}]);
        end
        return;
    end
    
    % Extract metrics for all algorithms
    pathLengths = [];
    durations = [];
    obstacleDistances = [];
    algorithmNames = {};

    for i = 1:length(algorithms)
        alg = algorithms{i};  % Using cell array indexing
        algData = results.(alg.fieldName);

        % Check if pathLengths exists and is not empty
        if isfield(algData, 'pathLengths') && ~isempty(algData.pathLengths)
            pathLengths = [pathLengths; mean(algData.pathLengths)];

            % Durations (required)
            if isfield(algData, 'globalPlanDurations') && ~isempty(algData.globalPlanDurations)
                durations = [durations; mean(algData.globalPlanDurations)];
            else
                durations = [durations; 0];
            end

            % Obstacle distances (optional - only available with hierarchical planning)
            if isfield(algData, 'obstacleDistances') && ~isempty(algData.obstacleDistances)
                obstacleDistances = [obstacleDistances; mean(algData.obstacleDistances)];
            else
                obstacleDistances = [obstacleDistances; NaN];  % Not available in direct planning
            end

            algorithmNames{end+1} = alg.displayName;
        end
    end
    
    % Display summary if we have enough data
    if length(pathLengths) >= 2
        fprintf('\n=== ALGORITHM COMPARISON SUMMARY ===\n');
        for i = 1:length(algorithmNames)
            if isnan(obstacleDistances(i))
                fprintf('%s: Path=%.2f, Time=%.4fs, Safety=N/A (direct planning)\n', ...
                    algorithmNames{i}, pathLengths(i), durations(i));
            else
                fprintf('%s: Path=%.2f, Time=%.4fs, Safety=%.2f\n', ...
                    algorithmNames{i}, pathLengths(i), durations(i), obstacleDistances(i));
            end
        end
    end
    
    disp('PSO comparison figures saved successfully!');
end

