function debug_critic_loss()
    % DEBUG_CRITIC_LOSS
    % Inspects critic loss data to understand why it's not plotting.

    clc;
    modelDir = fullfile('models', 'APEXPSO');
    filename = 'apexpso_perparticle_log.mat';
    filePath = fullfile(modelDir, filename);
    
    if ~exist(filePath, 'file')
        error('File not found: %s', filePath);
    end
    
    fprintf('Loading %s...\n', filename);
    loaded = load(filePath);
    data = loaded.logData;
    
    if isfield(data, 'critic1Loss')
        loss = data.critic1Loss;
        fprintf('Data size: %d x %d\n', size(loss));
        fprintf('First 10 values: ');
        fprintf('%.4f ', loss(1:10));
        fprintf('\n');
        
        fprintf('Min: %.4f, Max: %.4f\n', min(loss), max(loss));
        fprintf('Any NaN? %d\n', any(isnan(loss)));
        fprintf('Any Inf? %d\n', any(isinf(loss)));
        
        % Try simple plot
        f = figure('Visible', 'off');
        plot(loss);
        title('Debug Critic Loss');
        saveas(f, 'debug_critic_plot.png');
        fprintf('Saved debug_critic_plot.png\n');
    else
        fprintf('Field critic1Loss NOT found.\n');
        disp(fieldnames(data));
    end
end
