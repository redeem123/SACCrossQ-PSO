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

    requestedWorkers = parseRequestedWorkers();

    poolobj = gcp('nocreate');
    cluster = parcluster('local');

    if ~isempty(poolobj)
        if ~isempty(requestedWorkers) && poolobj.NumWorkers ~= requestedWorkers
            fprintf(['Existing parallel pool has %d workers but %d were requested. ', ...
                     'Restarting pool.\n'], poolobj.NumWorkers, requestedWorkers);
            delete(poolobj);
            poolobj = [];
        end
    end

    if isempty(poolobj)
        cleanupLocalClusterJobs(cluster);
        if isempty(requestedWorkers)
            % Default behavior: use all available physical cores.
            workerCount = feature('numcores');
            fprintf('Initializing parallel pool with %d workers (all available cores)...\n', workerCount);
        else
            workerCount = requestedWorkers;
            fprintf('Initializing parallel pool with %d workers (from VIETANH_PARPOOL_WORKERS)...\n', workerCount);
        end
        workerCount = clampWorkerCount(cluster, workerCount);
        poolobj = startPoolWithFallback(cluster, workerCount, requestedWorkers);
        fprintf('Parallel pool initialized successfully with %d workers.\n', poolobj.NumWorkers);
    else
        fprintf('Parallel pool already active with %d workers.\n', poolobj.NumWorkers);
    end

    if ~isempty(requestedWorkers) && poolobj.NumWorkers ~= requestedWorkers
        error(['Parallel pool worker count mismatch: expected %d workers from ', ...
               'VIETANH_PARPOOL_WORKERS but got %d.'], ...
               requestedWorkers, poolobj.NumWorkers);
    end

    try
        addAttachedFiles(poolobj, {'main.m'});
        fprintf('Added main.m to parallel pool workers.\n');
    catch
        % Optional attachment failure is ignored.
    end
end

function requestedWorkers = parseRequestedWorkers()
    requestedWorkers = [];
    rawValue = strtrim(getenv('VIETANH_PARPOOL_WORKERS'));
    if isempty(rawValue)
        return;
    end

    parsed = str2double(rawValue);
    if ~isfinite(parsed) || parsed <= 0 || floor(parsed) ~= parsed
        error('Invalid VIETANH_PARPOOL_WORKERS value: "%s". Expected a positive integer.', rawValue);
    end
    requestedWorkers = parsed;
end

function cleanupLocalClusterJobs(cluster)
    try
        jobs = cluster.Jobs;
        if isempty(jobs)
            return;
        end

        fprintf('Cleaning up %d stale local cluster job(s) before parpool.\n', numel(jobs));
        for i = 1:numel(jobs)
            try
                delete(jobs(i));
            catch
                % Best-effort cleanup; continue even if some old jobs are not owned
                % by this cluster instance anymore.
            end
        end
    catch
        % Best-effort only.
    end
end

function workerCount = clampWorkerCount(cluster, workerCount)
    try
        range = cluster.NumWorkersRange;
        if isnumeric(range) && numel(range) == 2
            workerCount = max(range(1), min(range(2), workerCount));
        end
    catch
        % Leave worker count unchanged when the profile does not expose a range.
    end
end

function poolobj = startPoolWithFallback(cluster, workerCount, requestedWorkers)
    try
        poolobj = parpool(cluster, workerCount);
        return;
    catch ME
        fprintf('Primary parpool startup failed: %s\n', ME.message);
        cleanupLocalClusterJobs(cluster);

        if ~isempty(requestedWorkers)
            fallbackWorkers = max(1, min(workerCount - 1, feature('numcores')));
            if fallbackWorkers ~= workerCount
                fprintf('Retrying parpool with %d workers after failure.\n', fallbackWorkers);
                poolobj = parpool(cluster, fallbackWorkers);
                return;
            end
        end

        rethrow(ME);
    end
end
