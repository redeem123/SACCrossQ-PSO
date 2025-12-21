function check_extra_metrics()
    modelDir = fullfile('models', 'APEXPSO');
    files = {'apexpso_perparticle_log.mat'}; % Check baseline
    
    for i = 1:length(files)
        file = files{i};
        path = fullfile(modelDir, file);
        if exist(path, 'file')
            d = load(path);
            log = d.logData;
            
            % Check Alpha
            if isfield(log, 'alphaValue')
                alpha = log.alphaValue;
                fprintf('Alpha: Mean=%.4f, Max=%.4f, Zeros=%d/%d\n', mean(alpha), max(alpha), sum(alpha==0), length(alpha));
            end
            
            % Check Entropy
            if isfield(log, 'policyEntropy')
                ent = log.policyEntropy;
                fprintf('Entropy: Mean=%.4f, Max=%.4f, Zeros=%d/%d\n', mean(ent), max(ent), sum(ent==0), length(ent));
            end
            
            % Check Q-Values
            if isfield(log, 'meanQValue')
                q = log.meanQValue;
                fprintf('Q-Value: Mean=%.4f, Max=%.4f, Zeros=%d/%d\n', mean(q), max(q), sum(q==0), length(q));
            end
        end
    end
end
