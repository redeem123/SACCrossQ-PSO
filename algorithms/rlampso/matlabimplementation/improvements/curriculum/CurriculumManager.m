classdef CurriculumManager < handle
    % Curriculum Learning Manager for Progressive Difficulty Training
    %
    % Manages automatic progression through difficulty levels based on
    % agent performance. Implements success rate tracking and auto-advancement.
    %
    % Difficulty Levels:
    %   Level 1 (Easy): Open space, no obstacles, distance <200
    %   Level 2 (Medium): 3-5 obstacles, 2-3 trees, distance 200-300
    %   Level 3 (Hard): 8-12 obstacles, 5-8 trees, distance 300-400
    %   Level 4 (Expert): 15+ obstacles, 10+ trees, complex terrain, distance >400

    properties
        currentLevel        % Current difficulty level (1-4)
        levels              % Cell array of level definitions
        successThreshold    % Success rate required to advance (default: 0.8)
        windowSize          % Episodes to track for success rate (default: 20)

        % Performance tracking
        episodeHistory      % History of episode outcomes (success/failure)
        successRates        % Success rate per level
        episodesPerLevel    % Episodes spent at each level
        currentLevelEpisodes % Episodes completed at current level

        % Statistics
        totalEpisodes       % Total training episodes
        levelTransitions    % Record of level changes
    end

    methods
        function obj = CurriculumManager(successThreshold, windowSize)
            % Constructor
            % Inputs:
            %   successThreshold: Success rate to advance (default: 0.8 = 80%)
            %   windowSize: Episodes to evaluate (default: 20)

            if nargin < 1, successThreshold = 0.8; end
            if nargin < 2, windowSize = 20; end

            obj.currentLevel = 1;
            obj.successThreshold = successThreshold;
            obj.windowSize = windowSize;

            % Initialize tracking
            obj.episodeHistory = [];
            obj.successRates = zeros(4, 1);
            obj.episodesPerLevel = zeros(4, 1);
            obj.currentLevelEpisodes = 0;
            obj.totalEpisodes = 0;
            obj.levelTransitions = [];

            % Define difficulty levels
            obj.levels = cell(4, 1);

            % Level 1: Easy - Open space
            obj.levels{1} = struct(...
                'name', 'Easy', ...
                'obstacleRange', [0, 2], ...
                'treeRange', [0, 1], ...
                'distanceRange', [100, 200], ...
                'terrainComplexity', 'flat', ...
                'description', 'Open space with minimal obstacles');

            % Level 2: Medium - Moderate obstacles
            obj.levels{2} = struct(...
                'name', 'Medium', ...
                'obstacleRange', [3, 5], ...
                'treeRange', [2, 3], ...
                'distanceRange', [200, 300], ...
                'terrainComplexity', 'gentle', ...
                'description', 'Moderate obstacles and terrain variation');

            % Level 3: Hard - Dense obstacles
            obj.levels{3} = struct(...
                'name', 'Hard', ...
                'obstacleRange', [8, 12], ...
                'treeRange', [5, 8], ...
                'distanceRange', [300, 400], ...
                'terrainComplexity', 'moderate', ...
                'description', 'Dense obstacles with complex terrain');

            % Level 4: Expert - Very challenging
            obj.levels{4} = struct(...
                'name', 'Expert', ...
                'obstacleRange', [15, 20], ...
                'treeRange', [10, 15], ...
                'distanceRange', [400, 500], ...
                'terrainComplexity', 'complex', ...
                'description', 'Maximum difficulty with complex terrain');

            fprintf('Curriculum Manager Initialized:\n');
            fprintf('  Starting Level: %d (%s)\n', obj.currentLevel, obj.levels{obj.currentLevel}.name);
            fprintf('  Success Threshold: %.1f%%\n', obj.successThreshold * 100);
            fprintf('  Evaluation Window: %d episodes\n', obj.windowSize);
        end

        function level = getCurrentLevel(obj)
            % Get current difficulty level definition
            level = obj.levels{obj.currentLevel};
        end

        function scenario = sampleScenario(obj)
            % Sample a scenario from current difficulty level
            scenario = generateCurriculumScenario(obj.levels{obj.currentLevel});
        end

        function recordEpisode(obj, success, fitness)
            % Record episode outcome and update statistics
            % Inputs:
            %   success: Boolean indicating episode success
            %   fitness: Final fitness value (lower is better)

            obj.totalEpisodes = obj.totalEpisodes + 1;
            obj.currentLevelEpisodes = obj.currentLevelEpisodes + 1;
            obj.episodesPerLevel(obj.currentLevel) = obj.episodesPerLevel(obj.currentLevel) + 1;

            % Add to history
            episode = struct('level', obj.currentLevel, 'success', success, 'fitness', fitness);
            obj.episodeHistory = [obj.episodeHistory, episode];

            % Update success rate for current level
            obj.updateSuccessRate();
        end

        function shouldAdvance = checkAdvancement(obj)
            % Check if agent should advance to next level
            % Returns true if success rate exceeds threshold and more levels exist

            shouldAdvance = false;

            % Need enough episodes at current level
            if obj.currentLevelEpisodes < obj.windowSize
                return;
            end

            % Check if at maximum level
            if obj.currentLevel >= length(obj.levels)
                return;
            end

            % Check success rate
            successRate = obj.successRates(obj.currentLevel);
            if successRate >= obj.successThreshold
                shouldAdvance = true;
            end
        end

        function advance(obj)
            % Advance to next difficulty level
            if obj.currentLevel >= length(obj.levels)
                warning('Already at maximum difficulty level');
                return;
            end

            oldLevel = obj.currentLevel;
            obj.currentLevel = obj.currentLevel + 1;
            obj.currentLevelEpisodes = 0;

            % Record transition
            transition = struct('episode', obj.totalEpisodes, ...
                              'from', oldLevel, ...
                              'to', obj.currentLevel, ...
                              'successRate', obj.successRates(oldLevel));
            obj.levelTransitions = [obj.levelTransitions, transition];

            fprintf('\n=== CURRICULUM ADVANCEMENT ===\n');
            fprintf('Advanced from Level %d (%s) to Level %d (%s)\n', ...
                    oldLevel, obj.levels{oldLevel}.name, ...
                    obj.currentLevel, obj.levels{obj.currentLevel}.name);
            fprintf('Success Rate: %.1f%% (threshold: %.1f%%)\n', ...
                    obj.successRates(oldLevel) * 100, obj.successThreshold * 100);
            fprintf('Total Episodes: %d\n', obj.totalEpisodes);
            fprintf('==============================\n\n');
        end

        function stats = getStatistics(obj)
            % Get comprehensive curriculum statistics
            stats = struct();
            stats.currentLevel = obj.currentLevel;
            stats.currentLevelName = obj.levels{obj.currentLevel}.name;
            stats.totalEpisodes = obj.totalEpisodes;
            stats.successRates = obj.successRates;
            stats.episodesPerLevel = obj.episodesPerLevel;
            stats.levelTransitions = obj.levelTransitions;
            stats.currentLevelProgress = obj.currentLevelEpisodes;
            stats.episodesToAdvancement = max(0, obj.windowSize - obj.currentLevelEpisodes);
        end

        function printProgress(obj)
            % Print current curriculum progress
            fprintf('\n--- Curriculum Progress ---\n');
            fprintf('Level: %d/%d (%s)\n', obj.currentLevel, length(obj.levels), ...
                    obj.levels{obj.currentLevel}.name);
            fprintf('Episodes at Level: %d\n', obj.currentLevelEpisodes);
            fprintf('Success Rate: %.1f%%\n', obj.successRates(obj.currentLevel) * 100);

            if obj.currentLevel < length(obj.levels)
                remaining = max(0, obj.windowSize - obj.currentLevelEpisodes);
                fprintf('Episodes Until Evaluation: %d\n', remaining);
                fprintf('Advancement Threshold: %.1f%%\n', obj.successThreshold * 100);
            else
                fprintf('STATUS: Maximum Difficulty Reached\n');
            end

            fprintf('--------------------------\n\n');
        end

        function save(obj, filename)
            % Save curriculum state to file
            curriculumState = struct();
            curriculumState.currentLevel = obj.currentLevel;
            curriculumState.episodeHistory = obj.episodeHistory;
            curriculumState.successRates = obj.successRates;
            curriculumState.episodesPerLevel = obj.episodesPerLevel;
            curriculumState.levelTransitions = obj.levelTransitions;
            curriculumState.totalEpisodes = obj.totalEpisodes;

            save(filename, 'curriculumState', '-v7.3');
            fprintf('Curriculum state saved to: %s\n', filename);
        end

        function load(obj, filename)
            % Load curriculum state from file
            if ~exist(filename, 'file')
                error('Curriculum state file not found: %s', filename);
            end

            loaded = load(filename);
            state = loaded.curriculumState;

            obj.currentLevel = state.currentLevel;
            obj.episodeHistory = state.episodeHistory;
            obj.successRates = state.successRates;
            obj.episodesPerLevel = state.episodesPerLevel;
            obj.levelTransitions = state.levelTransitions;
            obj.totalEpisodes = state.totalEpisodes;

            obj.currentLevelEpisodes = sum([obj.episodeHistory.level] == obj.currentLevel);

            fprintf('Curriculum state loaded from: %s\n', filename);
            fprintf('Resumed at Level %d (%s)\n', obj.currentLevel, obj.levels{obj.currentLevel}.name);
        end
    end

    methods (Access = private)
        function updateSuccessRate(obj)
            % Update success rate for current level using sliding window

            % Get episodes for current level
            levelEpisodes = obj.episodeHistory([obj.episodeHistory.level] == obj.currentLevel);

            if isempty(levelEpisodes)
                obj.successRates(obj.currentLevel) = 0;
                return;
            end

            % Use most recent windowSize episodes
            numEpisodes = length(levelEpisodes);
            startIdx = max(1, numEpisodes - obj.windowSize + 1);
            recentEpisodes = levelEpisodes(startIdx:end);

            % Calculate success rate
            numSuccesses = sum([recentEpisodes.success]);
            obj.successRates(obj.currentLevel) = numSuccesses / length(recentEpisodes);
        end
    end
end
