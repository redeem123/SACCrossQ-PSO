function scenarios = generateDiversePretrainingSet(numScenarios, mapSize)
    % Generate diverse set of scenarios for pretraining
    % Covers all difficulty levels to create robust base model
    %
    % Inputs:
    %   numScenarios: Number of scenarios to generate (default: 100)
    %   mapSize: Map dimensions [x, y, z] (default: [400, 400, 100])
    %
    % Outputs:
    %   scenarios: Cell array of diverse scenarios

    if nargin < 1, numScenarios = 100; end
    if nargin < 2, mapSize = [400, 400, 100]; end

    fprintf('Generating %d diverse pretraining scenarios...\n', numScenarios);

    scenarios = cell(numScenarios, 1);

    % Create curriculum manager to get level definitions
    tempCM = CurriculumManager();
    levels = tempCM.levels;

    % Distribute scenarios across difficulty levels
    scenariosPerLevel = floor(numScenarios / 4);
    extraScenarios = mod(numScenarios, 4);

    scenarioIdx = 1;

    % Generate scenarios for each difficulty level
    for levelIdx = 1:4
        numForLevel = scenariosPerLevel;

        % Add extra scenarios to higher difficulties
        if levelIdx <= extraScenarios
            numForLevel = numForLevel + 1;
        end

        fprintf('  Level %d (%s): %d scenarios\n', levelIdx, levels{levelIdx}.name, numForLevel);

        for i = 1:numForLevel
            scenarios{scenarioIdx} = generateCurriculumScenario(levels{levelIdx}, mapSize);
            scenarioIdx = scenarioIdx + 1;
        end
    end

    % Analyze diversity
    distances = zeros(numScenarios, 1);
    difficulties = zeros(numScenarios, 1);

    for i = 1:numScenarios
        distances(i) = scenarios{i}.actualDistance;
        difficulties(i) = scenarios{i}.difficulty;
    end

    fprintf('\nPretraining Set Statistics:\n');
    fprintf('  Distance: %.1f ± %.1f (range: [%.1f, %.1f])\n', ...
            mean(distances), std(distances), min(distances), max(distances));
    fprintf('  Difficulty: %.3f ± %.3f (range: [%.3f, %.3f])\n', ...
            mean(difficulties), std(difficulties), min(difficulties), max(difficulties));
    fprintf('  Total scenarios: %d\n', numScenarios);
    fprintf('✓ Diverse pretraining set generated\n\n');
end
