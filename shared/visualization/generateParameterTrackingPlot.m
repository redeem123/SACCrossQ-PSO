function generateParameterTrackingPlot(results, algorithms, scenarioIdx)
    % Generate parameter tracking plot for adaptive algorithms
    
    % Define which algorithms have parameter adaptation
    adaptiveAlgorithms = {
        'ActorCriticPSO', 'Actor-Critic PSO (DDPG)';
        'PGPSO', 'PG-PSO (Policy Gradient)';
        'PSO_TVAC', 'PSO-TVAC';
        'SAEPSO', 'SAEPSO (Full Implementation)';
        'APEXPSO_Online', 'APEX-PSO-Online (Online Learning)';
        'APEXPSO', 'APEX-PSO (SAC-CrossQ with Transformer)';
        'APEXPSO_Pretrained', 'APEX-PSO (Pretrained)';
        'APEXPSO_Run_Full', 'APEX-PSO Full (Baseline)';
        'APEXPSO_Abl_NoAtt', 'APEX-PSO No Attention';
        'APEXPSO_Abl_SimpleRwd', 'APEX-PSO Simple Reward';
        'APEXPSO_Abl_NoCrossQ', 'APEX-PSO No CrossQ';
    };
    
    % Count how many adaptive algorithms we have data for
    validAlgorithms = {};
    for i = 1:size(adaptiveAlgorithms, 1)
        algFieldName = adaptiveAlgorithms{i, 1};
        if isfield(results, algFieldName) && ...
           (isfield(results.(algFieldName), 'w_history') || ...
            isfield(results.(algFieldName), 'parameterHistory'))
            validAlgorithms{end+1, 1} = algFieldName;
            validAlgorithms{end, 2} = adaptiveAlgorithms{i, 2};
        end
    end
    
    if isempty(validAlgorithms)
        fprintf('No parameter adaptation data found for plotting.\n');
        return;
    end
    
    % Create figure with subplots for w, c1, c2
    h_fig = figure('Position', [100, 100, 1400, 1000], 'Color', 'w');
    
    % Colors for algorithms
    colors = [
        1.0, 0.0, 0.0;  % Red
        0.0, 0.0, 1.0;  % Blue  
        0.0, 0.8, 0.0;  % Green
        1.0, 0.5, 0.0;  % Orange
        0.5, 0.0, 0.5;  % Purple
        0.0, 0.8, 0.8;  % Cyan
        1.0, 0.0, 1.0;  % Magenta
    ];
    
    % Subplot 1: Inertia Weight (w)
    subplot(3, 1, 1);
    hold on;
    legendLabels = {};
    plotHandles = [];
    
    for i = 1:size(validAlgorithms, 1)
        algFieldName = validAlgorithms{i, 1};
        algDisplayName = validAlgorithms{i, 2};
        
        w_data = [];
        if isfield(results.(algFieldName), 'w_history')
            w_data = results.(algFieldName).w_history;
        elseif isfield(results.(algFieldName), 'parameterHistory') && isfield(results.(algFieldName).parameterHistory, 'w')
            w_data = results.(algFieldName).parameterHistory.w;
        end
        
        if ~isempty(w_data)
            iterations = 1:length(w_data);
            color = colors(mod(i-1, size(colors, 1)) + 1, :);
            h = plot(iterations, w_data, '-', 'Color', color, 'LineWidth', 2);
            plotHandles = [plotHandles, h];
            legendLabels{end+1} = algDisplayName;
        end
    end
    
    xlabel('Iteration', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Inertia Weight (w)', 'FontSize', 12, 'FontWeight', 'bold');
    % title(sprintf('Scenario %d: Parameter Adaptation - Inertia Weight', scenarioIdx), 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    if ~isempty(plotHandles)
        legend(plotHandles, legendLabels, 'Location', 'best', 'FontSize', 8);
    end
    set(gca, 'FontSize', 10);
    hold off;
    
    % Subplot 2: Cognitive Coefficient (c1)
    subplot(3, 1, 2);
    hold on;
    plotHandles = [];
    
    for i = 1:size(validAlgorithms, 1)
        algFieldName = validAlgorithms{i, 1};
        
        c1_data = [];
        if isfield(results.(algFieldName), 'c1_history')
            c1_data = results.(algFieldName).c1_history;
        elseif isfield(results.(algFieldName), 'parameterHistory') && isfield(results.(algFieldName).parameterHistory, 'c1')
            c1_data = results.(algFieldName).parameterHistory.c1;
        end
        
        if ~isempty(c1_data)
            iterations = 1:length(c1_data);
            color = colors(mod(i-1, size(colors, 1)) + 1, :);
            h = plot(iterations, c1_data, '-', 'Color', color, 'LineWidth', 2);
            plotHandles = [plotHandles, h];
        end
    end
    
    xlabel('Iteration', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Cognitive Coefficient (c1)', 'FontSize', 12, 'FontWeight', 'bold');
    % title(sprintf('Scenario %d: Parameter Adaptation - Cognitive Coefficient', scenarioIdx), 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    set(gca, 'FontSize', 10);
    hold off;
    
    % Subplot 3: Social Coefficient (c2)
    subplot(3, 1, 3);
    hold on;
    plotHandles = [];
    
    for i = 1:size(validAlgorithms, 1)
        algFieldName = validAlgorithms{i, 1};
        
        c2_data = [];
        if isfield(results.(algFieldName), 'c2_history')
            c2_data = results.(algFieldName).c2_history;
        elseif isfield(results.(algFieldName), 'parameterHistory') && isfield(results.(algFieldName).parameterHistory, 'c2')
            c2_data = results.(algFieldName).parameterHistory.c2;
        end
        
        if ~isempty(c2_data)
            iterations = 1:length(c2_data);
            color = colors(mod(i-1, size(colors, 1)) + 1, :);
            h = plot(iterations, c2_data, '-', 'Color', color, 'LineWidth', 2);
            plotHandles = [plotHandles, h];
        end
    end
    
    xlabel('Iteration', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Social Coefficient (c2)', 'FontSize', 12, 'FontWeight', 'bold');
    % title(sprintf('Scenario %d: Parameter Adaptation - Social Coefficient', scenarioIdx), 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    set(gca, 'FontSize', 10);
    hold off;
    
    % Adjust subplot spacing
    % sgtitle(sprintf('Parameter Evolution (Scenario %d)', scenarioIdx), 'FontSize', 16, 'FontWeight', 'bold');
    
    % Set up figure for PDF export
    set(gcf, 'PaperUnits', 'inches');
    set(gcf, 'PaperSize', [14, 10]);
    set(gcf, 'PaperPosition', [0, 0, 14, 10]);
    set(gcf, 'PaperPositionMode', 'manual');
    set(gcf, 'Renderer', 'painters');
    
    % Save to PDF
    resultsDir = getResultsDir();
    filename = fullfile(resultsDir, sprintf('Parameter_Tracking_Scenario%d', scenarioIdx));
    print(gcf, filename, '-dpdf', '-r300');

    fprintf('Parameter tracking plot saved as: %s.pdf\n', filename);
end

