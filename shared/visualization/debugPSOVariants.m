function debugPSOVariants(results, algorithms)
    % Helper function to debug metrics for PSO variants
    disp('==== PSO Debug Information ====');
    
    % Check if each algorithm exists
    for i = 1:length(algorithms)
        algName = algorithms{i}.fieldName;  % Using cell array indexing
        displayName = algorithms{i}.displayName;  % Using cell array indexing
        
        if isfield(results, algName)
            disp(['Algorithm ' displayName ' is present']);
            
            % Check core metrics
            coreMetrics = {'globalPlanTimes', 'globalPlanDurations', 'pathLengths', 'pathWaypoints', 'obstacleDistances'};
            for j = 1:length(coreMetrics)
                metricName = coreMetrics{j};
                if isfield(results.(algName), metricName)
                    metricData = results.(algName).(metricName);
                    if isempty(metricData)
                        disp(['  WARNING: ' metricName ' is empty for ' displayName]);
                    else
                        disp(['  ' metricName ': ' num2str(length(metricData)) ' data points']);
                        disp(['    Range: ' num2str(min(metricData)) ' to ' num2str(max(metricData))]);
                    end
                else
                    disp(['  ERROR: ' metricName ' is missing for ' displayName]);
                end
            end
        else
            disp(['ERROR: Algorithm ' displayName ' is not present in results']);
        end
        disp('-------------------');
    end
end

