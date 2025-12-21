function check_fitness()
    modelDir = fullfile('models', 'APEXPSO');
    files = {'apexpso_perparticle_log.mat', 'apexpso_abl_noatt_log.mat', 'apexpso_abl_simplerwd_log.mat', 'apexpso_abl_nocrossq_log.mat'};
    
    for i = 1:length(files)
        file = files{i};
        path = fullfile(modelDir, file);
        if exist(path, 'file')
            d = load(path);
            fit = d.logData.episodeFitness;
            fprintf('%s Fitness: Mean=%.2f, Min=%.2f, NonZero=%d/%d\n', ...
                file, mean(fit), min(fit), sum(fit~=0), length(fit));
        end
    end
end

