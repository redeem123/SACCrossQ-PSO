function check_all_losses()
    modelDir = fullfile('models', 'APEXPSO');
    files = {'apexpso_perparticle_log.mat', 'apexpso_abl_noatt_log.mat', 'apexpso_abl_simplerwd_log.mat', 'apexpso_abl_nocrossq_log.mat'};
    
    for i = 1:length(files)
        file = files{i};
        path = fullfile(modelDir, file);
        if exist(path, 'file')
            d = load(path);
            loss = d.logData.critic1Loss;
            fprintf('%s: Mean=%.4f, Max=%.4f, Zeros=%d/%d\n', ...
                file, mean(loss), max(loss), sum(loss==0), length(loss));
        end
    end
end

