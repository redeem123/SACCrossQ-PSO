function agent = initializeAgentAuto(config, pretrainedPath)
    % DL Toolbox Agent Initialization (REQUIRED)
    %
    % Uses MATLAB Deep Learning Toolbox (dlnetwork + automatic differentiation)
    % Will error if DL Toolbox is not available
    %
    % Inputs:
    %   config: Configuration from RLAMPSO_Config()
    %   pretrainedPath: Path to pretrained agent (optional)
    %
    % Outputs:
    %   agent: Initialized DL Toolbox agent
    %
    % Usage:
    %   config = RLAMPSO_Config('baseline');
    %   agent = initializeAgentAuto(config);

    if nargin < 2
        pretrainedPath = '';
    end

    % Only print if verbose mode enabled
    verbose = isfield(config, 'verbose') && config.verbose;

    if verbose
        fprintf('\n╔════════════════════════════════════════════╗\n');
        fprintf('║   Initializing RLAM-PSO Agent (DL Toolbox)║\n');
        fprintf('╚════════════════════════════════════════════╝\n\n');
    end

    % Verify DL Toolbox is available
    hasDLToolbox = checkDLToolboxAvailability();

    if ~hasDLToolbox
        error(['Deep Learning Toolbox is REQUIRED but not available!\n' ...
               'Please install Deep Learning Toolbox to use RLAMPSO.\n' ...
               'Run: matlab.addons.install() or install via Add-On Manager']);
    end

    % Use DL Toolbox implementation
    if verbose
        fprintf('✓ Deep Learning Toolbox detected\n');
        fprintf('→ Using DL Toolbox implementation (dlnetwork + automatic differentiation)\n\n');
    end

    if config.useTD3
        agent = initializeDLToolboxTD3(config.stateSize, config.actionSize, ...
                                       config.usePER, pretrainedPath, verbose);
    else
        agent = initializeDLToolboxDDPG(config.stateSize, config.actionSize, ...
                                        config.usePER, pretrainedPath, verbose);
    end

    % Display final configuration (only if verbose)
    if verbose
        fprintf('=== Agent Initialized ===\n');
        fprintf('Implementation: DL Toolbox (dlnetwork)\n');
        fprintf('Algorithm: %s\n', getAlgorithmType(agent));
        fprintf('State Size: %d | Action Size: %d\n', agent.stateSize, agent.actionSize);
        fprintf('PER: %s | GPU: %s\n', ...
                boolToStr(agent.usePER), boolToStr(agent.useGPU));
        fprintf('========================\n\n');
    end
end

%% Helper Functions

function available = checkDLToolboxAvailability()
    % Check if Deep Learning Toolbox is available
    try
        available = license('test', 'Neural_Network_Toolbox') && ...
                   exist('dlnetwork', 'file') == 2;
    catch
        available = false;
    end
end

function algType = getAlgorithmType(agent)
    % Get algorithm type string
    if isfield(agent, 'useTD3') && agent.useTD3
        algType = 'TD3 (Twin Delayed DDPG)';
    elseif isfield(agent, 'critic2')
        algType = 'TD3 (Twin Delayed DDPG)';
    else
        algType = 'DDPG';
    end
end

function str = boolToStr(value)
    if value
        str = 'ON';
    else
        str = 'OFF';
    end
end
