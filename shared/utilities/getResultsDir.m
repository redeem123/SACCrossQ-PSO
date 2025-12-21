function resultsDir = getResultsDir()
%GETRESULTSDIR Resolve (and create) the directory used for result exports.
%
%   Centralises path construction so the rest of the codebase does not rely
%   on hard-coded relative folders. The location is derived from the
%   repository root, assuming the structure initialised by run_comparison.m.

    persistent cachedDir

    if isempty(cachedDir) || ~isfolder(cachedDir)
        repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
        cachedDir = fullfile(repoRoot, 'outputs', 'results');
        if exist(cachedDir, 'dir') ~= 7
            mkdir(cachedDir);
        end
    end

    resultsDir = cachedDir;
end
