function generateAblationReport(ablationResults, configs, scenarios)
    % Generate detailed CSV report of ablation study
    
    timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    resultsDir = getResultsDir();
    filename = fullfile(resultsDir, sprintf('Ablation_Study_Report_%s.csv', timestamp));

    fid = fopen(filename, 'w');
    if fid == -1
        error('Could not open file for writing: %s', filename);
    end
    
    % Header
    fprintf(fid, 'Configuration,Scenario,Trees,Obstacles,MeanFitness,StdFitness,');
    fprintf(fid, 'MeanPathLength,StdPathLength,MeanTime,NeuralInfluence,');
    fprintf(fid, 'Architecture,FeatureSet,Training,AdaptiveInfluence,MaxGamma\n');
    
    % Data rows
    for s = 1:length(fieldnames(ablationResults))
        scenarioName = ['Scenario' num2str(s)];
        scenarioData = ablationResults.(scenarioName);
        
        for c = 1:length(configs)
            config = configs{c};
            configName = matlab.lang.makeValidName(config.name);
            
            if isfield(scenarioData, configName)
                results = scenarioData.(configName);
                
                fprintf(fid, '%s,%d,%d,%d,%.2f,%.2f,%.2f,%.2f,%.3f,%.3f,%s,%s,%s,%d,%.2f\n', ...
                    config.name, s, scenarios(s, 1), scenarios(s, 2), ...
                    results.meanFitness, results.stdFitness, ...
                    results.meanPathLength, results.stdPathLength, ...
                    results.meanExecutionTime, results.meanNeuralInfluence, ...
                    config.networkArchitecture, config.featureSet, ...
                    config.trainingStrategy, config.adaptiveInfluence, config.maxGamma);
            end
        end
    end
    
    fclose(fid);
    fprintf('Ablation report saved as: %s\n', filename);
end
