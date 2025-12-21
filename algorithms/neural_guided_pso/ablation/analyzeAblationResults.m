function analyzeAblationResults(ablationResults, configs, scenarios)
    % Analyze and visualize ablation study results
    
    fprintf('\n========== ABLATION STUDY ANALYSIS ==========\n\n');
    
    % Create comparison matrix
    numConfigs = length(configs);
    numScenarios = length(fieldnames(ablationResults));
    
    fitnessMatrix = zeros(numConfigs, numScenarios);
    pathLengthMatrix = zeros(numConfigs, numScenarios);
    timeMatrix = zeros(numConfigs, numScenarios);
    
    for s = 1:numScenarios
        scenarioName = ['Scenario' num2str(s)];
        scenarioData = ablationResults.(scenarioName);
        
        for c = 1:numConfigs
            configName = matlab.lang.makeValidName(configs{c}.name);
            if isfield(scenarioData, configName)
                fitnessMatrix(c, s) = scenarioData.(configName).meanFitness;
                pathLengthMatrix(c, s) = scenarioData.(configName).meanPathLength;
                timeMatrix(c, s) = scenarioData.(configName).meanExecutionTime;
            end
        end
    end
    
    % Calculate relative performance
    baselineIdx = 1;  % Baseline_NoNN
    fullIdx = 2;      % Full_NeuralGuidance
    
    for s = 1:numScenarios
        fprintf('=== Scenario %d Analysis ===\n', s);
        fprintf('%-30s | Fitness | Path Length | Time (s) | vs Baseline | vs Full\n', 'Configuration');
        fprintf('--------------------------------------------------------------------------------\n');
        
        for c = 1:numConfigs
            vsBaseline = ((fitnessMatrix(c, s) - fitnessMatrix(baselineIdx, s)) / ...
                         fitnessMatrix(baselineIdx, s)) * 100;
            vsFull = ((fitnessMatrix(c, s) - fitnessMatrix(fullIdx, s)) / ...
                     fitnessMatrix(fullIdx, s)) * 100;
            
            fprintf('%-30s | %7.1f | %11.1f | %8.2f | %+7.1f%% | %+7.1f%%\n', ...
                configs{c}.name, fitnessMatrix(c, s), pathLengthMatrix(c, s), ...
                timeMatrix(c, s), vsBaseline, vsFull);
        end
        fprintf('\n');
    end
    
    % Generate ablation plots
    generateAblationPlots(fitnessMatrix, configs, scenarios);
end

