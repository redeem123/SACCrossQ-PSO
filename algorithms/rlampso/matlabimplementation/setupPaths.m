function setupPaths()
    % SETUPPATHS Setup all MATLAB implementation paths
    %
    % This function adds all necessary MATLAB implementation directories
    % to the MATLAB path. Call this at the beginning of any script that
    % needs to access RLAMPSO components.
    %
    % Usage:
    %   setupPaths();
    %   config = RLAMPSO_Config('baseline');
    %   agent = initializeAgentAuto(config);

    % Get the directory where setupPaths.m is located (matlabimplementation root)
    matlabRoot = fileparts(mfilename('fullpath'));

    % Add main MATLAB implementation directory and all subdirectories
    addpath(matlabRoot);
    addpath(genpath(fullfile(matlabRoot, 'neural_networks')));
    addpath(genpath(fullfile(matlabRoot, 'benchmarks')));
    addpath(genpath(fullfile(matlabRoot, 'evaluation')));
    addpath(genpath(fullfile(matlabRoot, 'utils')));
    addpath(genpath(fullfile(matlabRoot, 'improvements')));
    addpath(genpath(fullfile(matlabRoot, 'training')));

    % Add shared environment functions (terrain, obstacles, trees)
    % Go up from matlabimplementation -> rlampso -> algorithms -> project root
    projectRoot = fileparts(fileparts(fileparts(matlabRoot)));
    addpath(genpath(fullfile(projectRoot, 'shared', 'environment')));
    addpath(genpath(fullfile(projectRoot, 'shared', 'utilities')));

    % Verify paths were added
    if ispc
        fprintf('✓ RLAMPSO MATLAB paths configured\n');
    end

end
