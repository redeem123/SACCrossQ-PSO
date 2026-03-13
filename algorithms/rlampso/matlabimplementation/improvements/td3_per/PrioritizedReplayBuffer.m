classdef PrioritizedReplayBuffer < handle
    % Prioritized Experience Replay Buffer with SumTree
    % Implements priority-based sampling for efficient learning
    %
    % Reference: Schaul et al., "Prioritized Experience Replay", ICLR 2016
    %
    % Key Features:
    %   - SumTree data structure for O(log n) operations
    %   - Priority-based sampling with importance weights
    %   - Alpha parameter for priority exponent (default: 0.6)
    %   - Beta annealing for importance sampling correction (0.4 → 1.0)

    properties
        capacity        % Maximum buffer size
        alpha           % Priority exponent (0 = uniform, 1 = full prioritization)
        beta            % Importance sampling weight (annealed to 1.0)
        beta_start      % Initial beta value
        beta_end        % Final beta value
        beta_increment  % Beta increment per sample
        epsilon         % Small constant to ensure non-zero priorities

        % SumTree properties
        tree            % Priority sum tree (size: 2*capacity - 1)
        data            % Experience storage (cell array)
        dataIndex       % Current write position
        size            % Current number of experiences
        maxPriority     % Track maximum priority for new experiences
    end

    methods
        function obj = PrioritizedReplayBuffer(capacity, alpha, beta_start, beta_frames)
            % Constructor
            % Inputs:
            %   capacity: Maximum buffer size
            %   alpha: Priority exponent (default: 0.6)
            %   beta_start: Initial importance sampling weight (default: 0.4)
            %   beta_frames: Number of frames to anneal beta to 1.0 (default: 100000)

            if nargin < 2, alpha = 0.6; end
            if nargin < 3, beta_start = 0.4; end
            if nargin < 4, beta_frames = 100000; end

            obj.capacity = capacity;
            obj.alpha = alpha;
            obj.beta_start = beta_start;
            obj.beta = beta_start;
            obj.beta_end = 1.0;
            obj.beta_increment = (obj.beta_end - beta_start) / beta_frames;
            obj.epsilon = 1e-6;

            % Initialize SumTree
            obj.tree = zeros(2 * capacity - 1, 1);
            obj.data = cell(capacity, 1);
            obj.dataIndex = 1;
            obj.size = 0;
            obj.maxPriority = 1.0;
        end

        function add(obj, experience, tdError)
            % Add experience to buffer with priority based on TD error
            % Inputs:
            %   experience: Struct with fields {state, action, reward, nextState, done}
            %   tdError: (optional) TD error for priority. If not provided, uses max priority

            if nargin < 3 || isempty(tdError)
                priority = obj.maxPriority;
            else
                priority = abs(tdError) + obj.epsilon;
                priority = priority ^ obj.alpha;
                obj.maxPriority = max(obj.maxPriority, priority);
            end

            % Store experience
            treeIndex = obj.dataIndex + obj.capacity - 1;
            obj.data{obj.dataIndex} = experience;

            % Update tree with new priority
            obj.updateTree(treeIndex, priority);

            % Move to next position
            obj.dataIndex = mod(obj.dataIndex, obj.capacity) + 1;
            obj.size = min(obj.size + 1, obj.capacity);
        end

        function [batch, indices, weights] = sample(obj, batchSize)
            % Sample batch using prioritized sampling
            % Outputs:
            %   batch: Cell array of experiences
            %   indices: Tree indices for updating priorities
            %   weights: Importance sampling weights

            batch = cell(batchSize, 1);
            indices = zeros(batchSize, 1);
            weights = zeros(batchSize, 1);

            % Get priority segments
            totalPriority = obj.tree(1);  % Root contains sum of all priorities
            segmentSize = totalPriority / batchSize;

            % Anneal beta
            obj.beta = min(obj.beta_end, obj.beta + obj.beta_increment);

            % Find minimum probability for importance weight normalization
            minPriority = min(obj.tree(obj.capacity:obj.capacity+obj.size-1));
            maxWeight = (minPriority / totalPriority * obj.size) ^ (-obj.beta);

            for i = 1:batchSize
                % Sample from priority distribution
                a = segmentSize * (i - 1);
                b = segmentSize * i;
                value = a + (b - a) * rand();

                % Retrieve sample index and priority
                [index, priority, dataIdx] = obj.get(value);

                % Calculate importance sampling weight
                samplingProb = priority / totalPriority;
                weight = (samplingProb * obj.size) ^ (-obj.beta);
                weights(i) = weight / maxWeight;  % Normalize by max weight

                indices(i) = index;
                batch{i} = obj.data{dataIdx};
            end
        end

        function update_priorities(obj, indices, tdErrors)
            % Update priorities for sampled experiences
            % Inputs:
            %   indices: Tree indices from sample()
            %   tdErrors: New TD errors for priority calculation

            for i = 1:length(indices)
                priority = (abs(tdErrors(i)) + obj.epsilon) ^ obj.alpha;
                obj.updateTree(indices(i), priority);
                obj.maxPriority = max(obj.maxPriority, priority);
            end
        end

        function len = length(obj)
            % Return current buffer size
            len = obj.size;
        end

        function tf = isFull(obj)
            % Check if buffer is full
            tf = (obj.size >= obj.capacity);
        end
    end

    methods (Access = private)
        function updateTree(obj, index, priority)
            % Update priority in SumTree and propagate changes
            % Tree structure: [parent nodes | leaf nodes]
            % Leaf nodes start at index: capacity

            change = priority - obj.tree(index);
            obj.tree(index) = priority;

            % Propagate change up the tree
            while index > 1
                index = floor((index - 1) / 2) + 1;  % Parent index (1-indexed)
                obj.tree(index) = obj.tree(index) + change;
            end
        end

        function [index, priority, dataIdx] = get(obj, value)
            % Retrieve experience index based on priority value
            % Uses binary search in SumTree

            parentIdx = 1;  % Start at root

            while true
                leftChildIdx = 2 * parentIdx;
                rightChildIdx = leftChildIdx + 1;

                % Check if leaf node
                if leftChildIdx >= length(obj.tree)
                    index = parentIdx;
                    break;
                end

                % Navigate tree based on cumulative priority
                if value <= obj.tree(leftChildIdx)
                    parentIdx = leftChildIdx;
                else
                    value = value - obj.tree(leftChildIdx);
                    parentIdx = rightChildIdx;
                end
            end

            % Convert tree index to data index
            dataIdx = index - obj.capacity + 1;

            % Handle circular buffer wrapping
            if dataIdx < 1 || dataIdx > obj.size
                dataIdx = mod(dataIdx - 1, obj.size) + 1;
            end

            priority = obj.tree(index);
        end
    end
end
