function [bestFitness, convergenceHistory, trainedAgent] = runRLAMPSOonCEC(functionID, dimensions, config, preloadedAgent)
    % Run RLAMPSO on CEC Benchmark Function
    %
    % This adapts RLAMPSO from UAV path planning to mathematical optimization
    % Same RL agent learns to adapt PSO parameters, different fitness function
    %
    % Inputs:
    %   functionID: CEC function number (1-28)
    %   dimensions: Problem dimensions (default: 30)
    %   config: RLAMPSO configuration from RLAMPSO_Config()
    %   preloadedAgent: Pre-initialized agent to reuse (default: empty)
    %
    % Outputs:
    %   bestFitness: Final best fitness value
    %   convergenceHistory: Fitness over iterations
    %   trainedAgent: Trained RL agent (for transfer learning)

    if nargin < 2, dimensions = 30; end
    if nargin < 3, config = RLAMPSO_Config('baseline'); end
    if nargin < 4, preloadedAgent = []; end

    % Get optimal value for convergence analysis
    [~, optimalValue] = CECBenchmarks(zeros(dimensions, 1), functionID);

    if isfield(config, 'verbose') && config.verbose
        fprintf('\n=== RLAMPSO on CEC Benchmark F%d (D=%d) ===\n', functionID, dimensions);
        fprintf('Optimal value: %.6f\n', optimalValue);
    end

    % ===== INITIALIZE AGENT =====
    if ~isempty(preloadedAgent)
        rlamAgent = preloadedAgent;
    else
        rlamAgent = initializeAgentAuto(config, '');
    end

    % Initialize Transformer State Encoder (if enabled)
    if config.useTransformerState
        % Feature dimension: position + velocity + fitness + metadata (3)
        inputFeatureDim = 2*dimensions + 4;
        transformerEncoder = initializeTransformerStateEncoder(inputFeatureDim, config.transformerHiddenDim, config.transformerNumHeads);
        temporalBuffer = [];
    else
        transformerEncoder = [];
        temporalBuffer = [];
    end

    % ===== INITIALIZE PSO PARAMETERS =====
    popSize = config.popSize;
    maxIterations = config.maxIterations;
    numSubgroups = 5;
    subgroupSize = floor(popSize / numSubgroups);

    % Search bounds (CEC benchmarks typically use [-100, 100])
    lowerBound = -100;
    upperBound = 100;

    % ===== INITIALIZE PARTICLES =====
    particles = struct();
    particles.cartesianPositions = lowerBound + (upperBound - lowerBound) * rand(popSize, dimensions);
    particles.velocities = (rand(popSize, dimensions) - 0.5) * 20;  % Initial velocity
    particles.bestPositions = particles.cartesianPositions;
    particles.bestFitness = Inf(popSize, 1);
    particles.fitness = Inf(popSize, 1);

    globalBestPosition = zeros(1, dimensions);
    globalBestFitness = Inf;
    previousGlobalBestFitness = Inf;
    lastImprovementIteration = 0;

    convergenceHistory = [];

    % Evaluate initial population
    for i = 1:popSize
        particles.fitness(i) = CECBenchmarks(particles.cartesianPositions(i, :)', functionID);
        particles.bestFitness(i) = particles.fitness(i);

        if particles.fitness(i) < globalBestFitness
            globalBestFitness = particles.fitness(i);
            globalBestPosition = particles.cartesianPositions(i, :);
            lastImprovementIteration = 0;
        end
    end

    convergenceHistory = [convergenceHistory; globalBestFitness];

    % ===== MAIN RLAMPSO LOOP =====
    previousReward = 0;
    previousNextState = [];

    for iter = 1:maxIterations
        previousGlobalBestFitness = globalBestFitness;

        % ===== ONLINE TRAINING (with warmup + reduced frequency) =====
        warmupPeriod = 50;
        trainingFrequency = 4;

        shouldTrain = iter > warmupPeriod && ...
                     ~isempty(previousNextState) && ...
                     mod(iter, trainingFrequency) == 0;

        if shouldTrain
            rlamAgent = updateAgentAuto(rlamAgent, previousNextState, [], previousReward);
        end

        % ===== CALCULATE STATE =====
        if config.useTransformerState
            [rlamState, ~] = calculateTransformerState(particles, iter, maxIterations, ...
                                                       lastImprovementIteration, ...
                                                       transformerEncoder, temporalBuffer);

            if size(temporalBuffer, 2) >= config.transformerTemporalWindow
                temporalBuffer = temporalBuffer(:, 2:end);
            end
            temporalBuffer = [temporalBuffer, rlamState];
        else
            rlamState = calculatePaperExactState(particles, iter, maxIterations, lastImprovementIteration);
        end

        % ===== GET ACTION =====
        [subgroupParams, rlamAgent] = getPaperExactAction(rlamAgent, rlamState, iter);

        % ===== UPDATE PARTICLES BY SUBGROUPS =====
        for subgroup = 1:numSubgroups
            startIdx = (subgroup - 1) * subgroupSize + 1;
            if subgroup == numSubgroups
                endIdx = popSize;
            else
                endIdx = subgroup * subgroupSize;
            end

            w_sub = subgroupParams(subgroup, 1);
            c1_sub = subgroupParams(subgroup, 2);
            c2_sub = subgroupParams(subgroup, 3);

            for i = startIdx:endIdx
                r1 = rand(1, dimensions);
                r2 = rand(1, dimensions);

                cognitiveComponent = c1_sub * r1 .* (particles.bestPositions(i, :) - particles.cartesianPositions(i, :));
                socialComponent = c2_sub * r2 .* (globalBestPosition - particles.cartesianPositions(i, :));

                particles.velocities(i, :) = w_sub * particles.velocities(i, :) + ...
                                             cognitiveComponent + socialComponent;

                % Velocity clamping
                maxVelocity = 0.2 * (upperBound - lowerBound);
                velocityMag = sqrt(sum(particles.velocities(i, :).^2));
                if velocityMag > maxVelocity
                    particles.velocities(i, :) = particles.velocities(i, :) * (maxVelocity / velocityMag);
                end

                % Update position
                particles.cartesianPositions(i, :) = particles.cartesianPositions(i, :) + particles.velocities(i, :);

                % Apply bounds
                particles.cartesianPositions(i, :) = max(lowerBound, min(upperBound, particles.cartesianPositions(i, :)));

                % Evaluate fitness
                particles.fitness(i) = CECBenchmarks(particles.cartesianPositions(i, :)', functionID);

                % Update personal best
                epsilon_update = 1e-6;
                if (particles.bestFitness(i) - particles.fitness(i)) > epsilon_update
                    particles.bestFitness(i) = particles.fitness(i);
                    particles.bestPositions(i, :) = particles.cartesianPositions(i, :);

                    % Update global best
                    if (globalBestFitness - particles.fitness(i)) > epsilon_update
                        globalBestFitness = particles.fitness(i);
                        globalBestPosition = particles.cartesianPositions(i, :);
                        lastImprovementIteration = iter;
                    end
                end
            end
        end

        % ===== CALCULATE REWARD (ADAPTIVE - TRANSFER LEARNING OPTIMIZED) =====
        % Use logarithmic scaling to handle huge dynamic range (10^-2 to 10^6)
        % This prevents saturation while preserving magnitude information

        fitnessImprovement = previousGlobalBestFitness - globalBestFitness;

        % Magnitude-aware reward with logarithmic scaling
        % BALANCED divisor: 4 (middle ground between 3 and 5)
        if abs(fitnessImprovement) > 1e-8
            if fitnessImprovement > 0
                % Improvement: positive reward scaled by log magnitude
                % log10(1 + improvement) ensures smooth scaling across orders of magnitude
                logMagnitude = log10(1 + abs(fitnessImprovement));
                reward = tanh(logMagnitude / 4);  % Divide by 4 for balanced learning
            else
                % Regression: negative reward
                logMagnitude = log10(1 + abs(fitnessImprovement));
                reward = -tanh(logMagnitude / 4);
            end
        else
            % No significant change
            reward = 0;
        end

        % Stagnation penalty (more aggressive for transfer learning)
        stagnationDuration = iter - lastImprovementIteration;
        if stagnationDuration > 20
            reward = reward - 0.3;  % Increased penalty
        elseif stagnationDuration > 50
            reward = -0.5;  % Strong penalty for prolonged stagnation
        end

        % Clamp reward to [-1, 1] to prevent extreme values
        reward = max(-1, min(1, reward));

        % ===== CALCULATE NEXT STATE =====
        if config.useTransformerState
            [nextState, ~] = calculateTransformerState(particles, iter + 1, maxIterations, ...
                                                       lastImprovementIteration, ...
                                                       transformerEncoder, temporalBuffer);
        else
            nextState = calculatePaperExactState(particles, iter + 1, maxIterations, lastImprovementIteration);
        end

        % ===== SAVE FOR NEXT ITERATION =====
        previousReward = reward;
        previousNextState = nextState;

        convergenceHistory = [convergenceHistory; globalBestFitness];

        % ===== PROGRESS REPORTING =====
        if isfield(config, 'verbose') && config.verbose
            if mod(iter, 50) == 0 || iter == 1 || iter == maxIterations
                errorToOptimal = abs(globalBestFitness - optimalValue);
                fprintf('[%d/%d] Fitness: %.6e | Error: %.6e | Stagnation: %d | Reward: %.2f\n', ...
                       iter, maxIterations, globalBestFitness, errorToOptimal, stagnationDuration, reward);
            end
        end

        % ===== EARLY TERMINATION =====
        % Success: Very close to optimal (1e-8 precision)
        if abs(globalBestFitness - optimalValue) < 1e-8
            if isfield(config, 'verbose') && config.verbose
                fprintf('✓ SUCCESS: Optimal solution found at iteration %d\n', iter);
            end
            break;
        end

        % Failure: Prolonged stagnation
        if stagnationDuration > 150
            if isfield(config, 'verbose') && config.verbose
                fprintf('⚠ STAGNATION: Terminating at iteration %d\n', iter);
            end
            break;
        end
    end

    % ===== FINAL TRAINING =====
    if ~isempty(previousNextState)
        rlamAgent = trainAgentAuto(rlamAgent, previousNextState, [], previousReward);
    end

    % ===== RETURN RESULTS =====
    bestFitness = globalBestFitness;
    trainedAgent = rlamAgent;

    if isfield(config, 'verbose') && config.verbose
        errorToOptimal = abs(globalBestFitness - optimalValue);
        fprintf('\n✓ Final Fitness: %.6e | Error to Optimal: %.6e\n', globalBestFitness, errorToOptimal);
    end
end
