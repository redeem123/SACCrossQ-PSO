function initializeParallelPool(executionMode)
    % Initialize parallel pool if not already active
    % executionMode: 'serial' or 'parallel'

    if nargin < 1 || isempty(executionMode)
        executionMode = 'serial';
    end

    executionMode = lower(strtrim(executionMode));
    if ~strcmp(executionMode, 'serial') && ~strcmp(executionMode, 'parallel')
        error('executionMode must be either ''serial'' or ''parallel''');
    end

    if strcmp(executionMode, 'serial')
        fprintf('Running in SERIAL mode (no parallel pool)\n');
        return;
    end

    fprintf('Running in PARALLEL mode\n');

    try
        poolobj = gcp('nocreate');
        if isempty(poolobj)
            cluster = parcluster('local');
            % Use all available physical cores
            numCores = feature('numcores');
            maxWorkers = numCores;  % Use all cores (APEXPSO runs serially anyway)
            fprintf('Initializing parallel pool with %d workers (all available cores)...\n', maxWorkers);
            poolobj = parpool(cluster, maxWorkers);
            fprintf('Parallel pool initialized successfully with %d workers.\n', poolobj.NumWorkers);

            try
                addAttachedFiles(poolobj, {'main.m'});
                fprintf('Added main.m to parallel pool workers.\n');
            catch
                % Optional attachment failure is ignored
            end
        else
            fprintf('Parallel pool already active with %d workers.\n', poolobj.NumWorkers);
            try
                addAttachedFiles(poolobj, {'main.m'});
                fprintf('Added main.m to existing parallel pool workers.\n');
            catch
                % Optional attachment failure is ignored
            end
        end
    catch ME
        warning('Failed to initialize parallel pool: %s. Running in serial mode.', ME.message);
    end
end
