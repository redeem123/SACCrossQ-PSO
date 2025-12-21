classdef ReplayBuffer < handle
    % Experience Replay Buffer for SAC
    %
    % Stores transitions (state, action, reward, next_state, done) in a circular buffer
    % for off-policy learning.
    %
    % Features:
    %   - Fixed-size circular buffer
    %   - Random batch sampling
    %   - GPU support (optional)

    properties
        capacity        % Maximum buffer size
        size            % Current number of stored experiences
        position        % Current write position (circular)

        % Storage arrays
        states          % [capacity × stateSize]
        actions         % [capacity × actionSize]
        rewards         % [capacity × 1]
        nextStates      % [capacity × stateSize]
        dones           % [capacity × 1] (episode terminal flags)

        stateSize       % Dimension of state
        actionSize      % Dimension of action
        useGPU          % Whether to store on GPU
    end

    methods
        function obj = ReplayBuffer(capacity, stateSize, actionSize, useGPU)
            % Constructor
            %
            % Inputs:
            %   capacity: Maximum buffer size
            %   stateSize: Dimension of state vector
            %   actionSize: Dimension of action vector
            %   useGPU: Store on GPU (optional)

            if nargin < 4
                useGPU = false;
            end

            obj.capacity = capacity;
            obj.stateSize = stateSize;
            obj.actionSize = actionSize;
            obj.useGPU = useGPU && gpuDeviceCount > 0;

            obj.size = 0;
            obj.position = 0;

            % Pre-allocate storage
            if obj.useGPU
                obj.states = gpuArray(zeros(capacity, stateSize, 'single'));
                obj.actions = gpuArray(zeros(capacity, actionSize, 'single'));
                obj.rewards = gpuArray(zeros(capacity, 1, 'single'));
                obj.nextStates = gpuArray(zeros(capacity, stateSize, 'single'));
                obj.dones = gpuArray(zeros(capacity, 1, 'single'));
            else
                obj.states = zeros(capacity, stateSize);
                obj.actions = zeros(capacity, actionSize);
                obj.rewards = zeros(capacity, 1);
                obj.nextStates = zeros(capacity, stateSize);
                obj.dones = zeros(capacity, 1);
            end
        end

        function add(obj, state, action, reward, nextState, done)
            % Add a transition to the buffer
            %
            % Inputs:
            %   state: Current state [stateSize × 1]
            %   action: Action taken [actionSize × 1]
            %   reward: Reward received (scalar)
            %   nextState: Next state [stateSize × 1]
            %   done: Episode done flag (0 or 1)

            obj.position = mod(obj.position, obj.capacity) + 1;

            obj.states(obj.position, :) = state';
            obj.actions(obj.position, :) = action';
            obj.rewards(obj.position) = reward;
            obj.nextStates(obj.position, :) = nextState';
            obj.dones(obj.position) = done;

            obj.size = min(obj.size + 1, obj.capacity);
        end

        function [states, actions, rewards, nextStates, dones] = sample(obj, batchSize)
            % Sample a random batch of transitions
            %
            % Inputs:
            %   batchSize: Number of transitions to sample
            %
            % Outputs:
            %   states: [batchSize × stateSize]
            %   actions: [batchSize × actionSize]
            %   rewards: [batchSize × 1]
            %   nextStates: [batchSize × stateSize]
            %   dones: [batchSize × 1]

            % Random indices
            indices = randi(obj.size, batchSize, 1);

            % Extract batch
            states = obj.states(indices, :);
            actions = obj.actions(indices, :);
            rewards = obj.rewards(indices);
            nextStates = obj.nextStates(indices, :);
            dones = obj.dones(indices);
        end

        function canSample = canSample(obj, batchSize)
            % Check if buffer has enough samples
            canSample = obj.size >= batchSize;
        end

        function clear(obj)
            % Clear the buffer
            obj.size = 0;
            obj.position = 0;
        end

        function toCPU(obj)
            % Convert all buffer data to CPU arrays
            if obj.useGPU
                obj.states = gather(obj.states);
                obj.actions = gather(obj.actions);
                obj.rewards = gather(obj.rewards);
                obj.nextStates = gather(obj.nextStates);
                obj.dones = gather(obj.dones);
                obj.useGPU = false;
            end
        end
    end
end
