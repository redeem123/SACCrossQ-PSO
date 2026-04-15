function agent = initializeAgentAuto(config, pretrainedPath)
    % Initialize DDPG agent using DL Toolbox (paper-exact)
    %
    % Requires MATLAB Deep Learning Toolbox (dlnetwork).

    if nargin < 2, pretrainedPath = ''; end

    verbose = isfield(config, 'verbose') && config.verbose;

    % Verify DL Toolbox
    hasDL = false;
    try
        hasDL = license('test', 'Neural_Network_Toolbox') && exist('dlnetwork', 'file') == 2;
    catch
    end
    if ~hasDL
        error('Deep Learning Toolbox is REQUIRED but not available.');
    end

    if verbose
        fprintf('Initializing RLAM-PSO DDPG Agent (DL Toolbox)\n');
    end

    agent = initializeDLToolboxDDPG(config.stateSize, config.actionSize, ...
                                     false, pretrainedPath, verbose);
end
