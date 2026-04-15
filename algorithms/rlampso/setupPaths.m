function setupPaths()
    % SETUPPATHS Setup RLAMPSO paths (flat directory, no subdirectories)

    rlampsoRoot = fileparts(mfilename('fullpath'));
    addpath(rlampsoRoot);

    % Go up from rlampso -> algorithms -> project root
    projectRoot = fileparts(fileparts(rlampsoRoot));
    addpath(genpath(fullfile(projectRoot, 'shared', 'environment')));
    addpath(genpath(fullfile(projectRoot, 'shared', 'evaluation')));
    addpath(genpath(fullfile(projectRoot, 'shared', 'utilities')));
    addpath(genpath(fullfile(projectRoot, 'shared', 'benchmarks')));
end
