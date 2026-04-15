classdef ReplayBuffer < handle
    % Experience Replay Buffer for SAC
    %
    % Stores transitions (state, action, reward, next_state, done) in a circular buffer
    % for off-policy learning.
    %
    % Features:
    %   - Fixed-size circular buffer
    %   - Optional prioritized replay
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
        discounts       % [capacity × 1] one-step bootstrap discount multiplier
        auxRewards      % [capacity × 1] auxiliary long-horizon reward sum
        auxNextStates   % [capacity × stateSize] auxiliary long-horizon next states
        auxDones        % [capacity × 1] auxiliary long-horizon terminal flags
        auxDiscounts    % [capacity × 1] auxiliary long-horizon discount multiplier

        stateSize       % Dimension of state
        actionSize      % Dimension of action
        useGPU          % Whether to store on GPU
        usePrioritizedReplay
        usePilarReturns
        pilarLongHorizon
        gamma
        priorityAlpha
        priorityEpsilon
        maxPriority
        priorities
        returnQueue
    end

    methods
        function obj = ReplayBuffer(capacity, stateSize, actionSize, useGPU, config)
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
            if nargin < 5 || isempty(config)
                config = struct();
            end

            obj.capacity = capacity;
            obj.stateSize = stateSize;
            obj.actionSize = actionSize;
            obj.useGPU = useGPU && gpuDeviceCount > 0;
            obj.usePrioritizedReplay = nargin >= 5 && isstruct(config) && ...
                isfield(config, 'usePrioritizedReplay') && config.usePrioritizedReplay;
            obj.usePilarReturns = nargin >= 5 && isstruct(config) && ...
                isfield(config, 'usePilarReturns') && config.usePilarReturns;
            obj.pilarLongHorizon = max(1, round(getConfigValue(config, 'pilarLongHorizon', 1)));
            obj.gamma = getConfigValue(config, 'gamma', 0.99);
            obj.priorityAlpha = getConfigValue(config, 'priorityReplayAlpha', 0.6);
            obj.priorityEpsilon = getConfigValue(config, 'priorityReplayEpsilon', 1e-3);
            obj.maxPriority = 1.0;
            obj.returnQueue = struct('state', {}, 'action', {}, 'reward', {}, 'nextState', {}, 'done', {});

            obj.size = 0;
            obj.position = 0;

            % Pre-allocate storage
            if obj.useGPU
                obj.states = gpuArray(zeros(capacity, stateSize, 'single'));
                obj.actions = gpuArray(zeros(capacity, actionSize, 'single'));
                obj.rewards = gpuArray(zeros(capacity, 1, 'single'));
                obj.nextStates = gpuArray(zeros(capacity, stateSize, 'single'));
                obj.dones = gpuArray(zeros(capacity, 1, 'single'));
                obj.discounts = gpuArray(ones(capacity, 1, 'single') * single(obj.gamma));
                obj.auxRewards = gpuArray(zeros(capacity, 1, 'single'));
                obj.auxNextStates = gpuArray(zeros(capacity, stateSize, 'single'));
                obj.auxDones = gpuArray(zeros(capacity, 1, 'single'));
                obj.auxDiscounts = gpuArray(ones(capacity, 1, 'single') * single(obj.gamma));
                obj.priorities = gpuArray(ones(capacity, 1, 'single'));
            else
                obj.states = zeros(capacity, stateSize);
                obj.actions = zeros(capacity, actionSize);
                obj.rewards = zeros(capacity, 1);
                obj.nextStates = zeros(capacity, stateSize);
                obj.dones = zeros(capacity, 1);
                obj.discounts = ones(capacity, 1) * obj.gamma;
                obj.auxRewards = zeros(capacity, 1);
                obj.auxNextStates = zeros(capacity, stateSize);
                obj.auxDones = zeros(capacity, 1);
                obj.auxDiscounts = ones(capacity, 1) * obj.gamma;
                obj.priorities = ones(capacity, 1);
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

            if obj.usePilarReturns && obj.pilarLongHorizon > 1
                transition = struct( ...
                    'state', state(:), ...
                    'action', action(:), ...
                    'reward', reward, ...
                    'nextState', nextState(:), ...
                    'done', logical(done));
                obj.returnQueue(end + 1) = transition;
                obj.emitPilarTransitions(logical(done));
                return;
            end

            obj.storeTransition(state, action, reward, nextState, done, obj.gamma, ...
                reward, nextState, done, obj.gamma);
        end

        function [states, actions, rewards, nextStates, dones, discounts, indices, weights, ...
                auxRewards, auxNextStates, auxDones, auxDiscounts] = sample(obj, batchSize, beta, recentWindowSize)
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

            if nargin < 3
                beta = 0.4;
            end
            if nargin < 4 || isempty(recentWindowSize)
                recentWindowSize = obj.size;
            end

            candidateIndices = obj.getRecentIndices(recentWindowSize);
            candidateCount = numel(candidateIndices);
            if candidateCount == 0
                error('ReplayBuffer:EmptySampleWindow', 'No samples available in the requested replay window.');
            end

            if obj.usePrioritizedReplay
                activePriorities = obj.priorities(candidateIndices);
                scaledPriorities = activePriorities .^ obj.priorityAlpha;
                totalPriority = sum(scaledPriorities);
                if totalPriority <= 0 || ~isfinite(gather(totalPriority))
                    probabilities = ones(candidateCount, 1, 'like', activePriorities) / candidateCount;
                else
                    probabilities = scaledPriorities / totalPriority;
                end

                cdf = cumsum(gather(probabilities));
                cdf(end) = 1;
                samples = rand(batchSize, 1);
                localIndices = arrayfun(@(x) find(cdf >= x, 1, 'first'), samples, 'UniformOutput', true);
                indices = candidateIndices(localIndices);
                sampleProbabilities = max(gather(probabilities(localIndices)), eps('single'));
                weights = (candidateCount .* sampleProbabilities) .^ (-beta);
                weights = weights / max(weights);
                weights = single(weights);
            else
                localIndices = randi(candidateCount, batchSize, 1);
                indices = candidateIndices(localIndices);
                weights = ones(batchSize, 1, 'single');
            end

            % Extract batch
            states = obj.states(indices, :);
            actions = obj.actions(indices, :);
            rewards = obj.rewards(indices);
            nextStates = obj.nextStates(indices, :);
            dones = obj.dones(indices);
            discounts = obj.discounts(indices);
            auxRewards = obj.auxRewards(indices);
            auxNextStates = obj.auxNextStates(indices, :);
            auxDones = obj.auxDones(indices);
            auxDiscounts = obj.auxDiscounts(indices);
        end

        function updatePriorities(obj, indices, priorities)
            if ~obj.usePrioritizedReplay || isempty(indices)
                return;
            end

            priorities = gather(priorities(:));
            priorities = max(priorities, obj.priorityEpsilon);
            obj.maxPriority = max(obj.maxPriority, max(priorities));
            obj.priorities(indices) = cast(priorities, 'like', obj.priorities);
        end

        function canSample = canSample(obj, batchSize)
            % Check if buffer has enough samples
            canSample = obj.size >= batchSize;
        end

        function clear(obj)
            % Clear the buffer
            obj.size = 0;
            obj.position = 0;
            obj.maxPriority = 1.0;
            obj.priorities(:) = 1;
            obj.discounts(:) = obj.gamma;
            obj.auxDiscounts(:) = obj.gamma;
            obj.returnQueue = struct('state', {}, 'action', {}, 'reward', {}, 'nextState', {}, 'done', {});
        end

        function toCPU(obj)
            % Convert all buffer data to CPU arrays
            if obj.useGPU
                obj.states = gather(obj.states);
                obj.actions = gather(obj.actions);
                obj.rewards = gather(obj.rewards);
                obj.nextStates = gather(obj.nextStates);
                obj.dones = gather(obj.dones);
                obj.discounts = gather(obj.discounts);
                obj.auxRewards = gather(obj.auxRewards);
                obj.auxNextStates = gather(obj.auxNextStates);
                obj.auxDones = gather(obj.auxDones);
                obj.auxDiscounts = gather(obj.auxDiscounts);
                obj.priorities = gather(obj.priorities);
                obj.useGPU = false;
            end
        end

        function emitPilarTransitions(obj, flushRemainder)
            if nargin < 2
                flushRemainder = false;
            end

            while ~isempty(obj.returnQueue) && ...
                    (numel(obj.returnQueue) >= obj.pilarLongHorizon || flushRemainder)
                stepsToUse = min(obj.pilarLongHorizon, numel(obj.returnQueue));
                longRewardSum = 0;
                longDiscountPower = 1;
                terminalReached = false;
                longNextState = obj.returnQueue(stepsToUse).nextState;
                actualSteps = 0;

                for stepIdx = 1:stepsToUse
                    longRewardSum = longRewardSum + longDiscountPower * obj.returnQueue(stepIdx).reward;
                    actualSteps = stepIdx;
                    longNextState = obj.returnQueue(stepIdx).nextState;
                    if obj.returnQueue(stepIdx).done
                        terminalReached = true;
                        break;
                    end
                    longDiscountPower = longDiscountPower * obj.gamma;
                end

                longBootstrapDiscount = obj.gamma ^ actualSteps;
                obj.storeTransition( ...
                    obj.returnQueue(1).state, ...
                    obj.returnQueue(1).action, ...
                    obj.returnQueue(1).reward, ...
                    obj.returnQueue(1).nextState, ...
                    double(obj.returnQueue(1).done), ...
                    obj.gamma, ...
                    longRewardSum, ...
                    longNextState, ...
                    double(terminalReached), ...
                    longBootstrapDiscount);
                obj.returnQueue(1) = [];

                if ~flushRemainder && numel(obj.returnQueue) < obj.pilarLongHorizon
                    break;
                end
            end
        end
    end

    methods (Access = private)
        function indices = getRecentIndices(obj, recentWindowSize)
            if obj.size == 0
                indices = zeros(0, 1);
                return;
            end

            recentWindowSize = max(1, min(obj.size, round(recentWindowSize)));
            startPosition = obj.position - recentWindowSize + 1;
            indices = mod((startPosition:obj.position) - 1, obj.capacity) + 1;
            indices = indices(:);
        end

        function storeTransition(obj, state, action, reward, nextState, done, discount, ...
                auxReward, auxNextState, auxDone, auxDiscount)
            obj.position = mod(obj.position, obj.capacity) + 1;

            obj.states(obj.position, :) = state(:)';
            obj.actions(obj.position, :) = action(:)';
            obj.rewards(obj.position) = reward;
            obj.nextStates(obj.position, :) = nextState(:)';
            obj.dones(obj.position) = done;
            obj.discounts(obj.position) = discount;
            obj.auxRewards(obj.position) = auxReward;
            obj.auxNextStates(obj.position, :) = auxNextState(:)';
            obj.auxDones(obj.position) = auxDone;
            obj.auxDiscounts(obj.position) = auxDiscount;
            if obj.usePrioritizedReplay
                obj.priorities(obj.position) = obj.maxPriority;
            end
            obj.size = min(obj.size + 1, obj.capacity);
        end
    end
end

function value = getConfigValue(config, fieldName, defaultValue)
    if nargin >= 1 && isstruct(config) && isfield(config, fieldName)
        value = config.(fieldName);
    else
        value = defaultValue;
    end
end
