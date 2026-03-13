function resultsDir = getResultsDir()
%GETRESULTSDIR Resolve (and create) the directory used for result exports.
%
%   Centralises path construction so the rest of the codebase does not rely
%   on hard-coded relative folders. The location is derived from the
%   repository root, assuming the structure initialised by run_comparison.m.

    persistent cachedDir

    envOverride = strtrim(getenv('VIETANH_RESULTS_DIR'));

    if ~isempty(envOverride)
        if ensureWritableDirectory(envOverride)
            cachedDir = envOverride;
        else
            error('VIETANH_RESULTS_DIR is not writable: %s', envOverride);
        end
    elseif isempty(cachedDir) || ~isfolder(cachedDir) || ~ensureWritableDirectory(cachedDir)
        repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
        primaryDir = fullfile(repoRoot, 'outputs', 'results');
        if ensureWritableDirectory(primaryDir)
            cachedDir = primaryDir;
        else
            fallbackDir = fullfile(tempdir, 'vietanhpaper-2', 'results');
            if ~ensureWritableDirectory(fallbackDir)
                error('Unable to create a writable results directory. Tried: %s and %s', primaryDir, fallbackDir);
            end
            fprintf('Results directory not writable: %s\n', primaryDir);
            fprintf('Writing results to fallback: %s\n', fallbackDir);
            cachedDir = fallbackDir;
        end
    end

    resultsDir = cachedDir;
end

function ok = ensureWritableDirectory(directoryPath)
    ok = true;
    if exist(directoryPath, 'dir') ~= 7
        [ok, msg] = mkdir(directoryPath);
        if ~ok
            fprintf('Failed to create directory: %s\n', directoryPath);
            fprintf('Reason: %s\n', msg);
            return;
        end
    end

    testFile = fullfile(directoryPath, sprintf('.writetest_%s', ...
        char(datetime('now', 'Format', 'yyyyMMdd_HHmmssSSS'))));
    fid = fopen(testFile, 'w');
    if fid == -1
        ok = false;
        return;
    end
    fclose(fid);
    delete(testFile);
end
