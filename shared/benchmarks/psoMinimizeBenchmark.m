function [bestX, bestFitness, fitnessHistory] = psoMinimizeBenchmark( ...
        objFunc, D, lb, ub, agent, stateEncoder, config)
%PSOMINIMIZEBENCHMARK Generic PSO with SAC parameter adaptation for benchmarks.
%
%   Supports two modes via config.stateMode:
%     'rich' (default)  — AFSACPSO 15D state + attractor-field 9D action
%     'paper'           — SAC-SAPSO paper state (ns+3) + global 3D action
%
%   Inputs:
%     objFunc       — function handle @(x) where x is 1×D
%     D             — dimensionality
%     lb, ub        — 1×D bounds
%     agent         — AFSACPSO_Agent instance (shared across episodes)
%     stateEncoder  — SimpleStateEncoder instance (ignored in paper mode)
%     config        — AFSACPSO config struct
%
%   Outputs:
%     bestX         — 1×D best solution found
%     bestFitness   — scalar best objective value
%     fitnessHistory — 1×maxIter convergence trace

    popSize = config.popSize;
    maxIter = config.maxIterations;

    % Mode flags
    usePaperState = isfield(config, 'stateMode') && strcmp(config.stateMode, 'paper');
    usePaperReward = isfield(config, 'rewardMode') && strcmp(config.rewardMode, 'paper');
    observationInterval = 1;
    if isfield(config, 'observationInterval')
        observationInterval = config.observationInterval;
    end

    % Velocity clamping
    delta = 0.15;
    if isfield(config, 'velocityClampDelta')
        delta = config.velocityClampDelta;
    end
    vMax = delta * (ub - lb);

    % Store bounds in config for buildPaperState
    config.benchmarkLB = lb;
    config.benchmarkUB = ub;

    % --- Initialize particles ---
    particles = struct();
    particles.cartesianPositions = lb + rand(popSize, D) .* (ub - lb);
    particles.velocities = zeros(popSize, D);
    particles.fitness = inf(popSize, 1);
    particles.bestPositions = particles.cartesianPositions;
    particles.bestFitness = inf(popSize, 1);
    particles.stagnationCounter = zeros(popSize, 1);
    particles.isFeasible = true(popSize, 1);

    % Evaluate initial population
    for i = 1:popSize
        particles.fitness(i) = objFunc(particles.cartesianPositions(i,:));
        particles.bestFitness(i) = particles.fitness(i);
    end

    [episodeBestFitness, ~] = min(particles.bestFitness);
    lastImprovementIter = 0;
    fitnessHistory = zeros(1, maxIter);

    % --- Build initial state ---
    if usePaperState
        config.lastParticleParams = [];
        state = buildPaperState(particles, 1, maxIter, config);
    else
        stateEncoder.reset();
        [initFeatures, particles] = extractRichFeatures(particles, 1, maxIter, 0, episodeBestFitness);
        stateEncoder.addIteration(initFeatures);
        state = stateEncoder.encode();
    end

    % --- Get initial action ---
    if isfield(config, 'forcedZeroAction') && config.forcedZeroAction
        action = zeros(1, config.actionSize);
    else
        action = agent.getAction(state, true);
    end
    particleParams = convertActionToPerParticleParams(action, config, particles, 1, maxIter);
    config.lastParticleParams = particleParams;

    % For paper reward: track fitness at last observation
    lastObsFitness = episodeBestFitness;

    % --- PSO loop ---
    for iter = 1:maxIter
        % Paper mode: only observe/act every nt iterations
        shouldObserve = (observationInterval <= 1) || ...
            (mod(iter, observationInterval) == 0) || (iter == maxIter);

        if shouldObserve && iter > 1
            % Get new action
            if isfield(config, 'forcedZeroAction') && config.forcedZeroAction
                action = zeros(1, config.actionSize);
            else
                action = agent.getAction(state, true);
            end
            particleParams = convertActionToPerParticleParams(action, config, particles, iter, maxIter);
            config.lastParticleParams = particleParams;
        end

        % Update particles using current particleParams
        [~, gbestIdx] = min(particles.bestFitness);
        gbestPos = particles.bestPositions(gbestIdx, :);

        for i = 1:popSize
            w  = particleParams(i,1);
            c1 = particleParams(i,2);
            c2 = particleParams(i,3);

            r1 = rand(1, D);
            r2 = rand(1, D);
            particles.velocities(i,:) = w * particles.velocities(i,:) ...
                + c1 * r1 .* (particles.bestPositions(i,:) - particles.cartesianPositions(i,:)) ...
                + c2 * r2 .* (gbestPos - particles.cartesianPositions(i,:));

            % Per-dimension velocity clamping
            particles.velocities(i,:) = max(-vMax, min(vMax, particles.velocities(i,:)));

            % Position update
            particles.cartesianPositions(i,:) = particles.cartesianPositions(i,:) ...
                + particles.velocities(i,:);

            % Boundary handling: paper excludes infeasible from best updates
            inBounds = all(particles.cartesianPositions(i,:) >= lb) && ...
                       all(particles.cartesianPositions(i,:) <= ub);
            particles.isFeasible(i) = inBounds;

            if inBounds
                % Clamp to bounds (for position, keep in feasible region)
                particles.cartesianPositions(i,:) = max(lb, min(ub, particles.cartesianPositions(i,:)));
                particles.fitness(i) = objFunc(particles.cartesianPositions(i,:));
            else
                % Paper: infeasible particles get infinite fitness
                particles.fitness(i) = inf;
            end

            % Update personal best (only if feasible)
            if particles.fitness(i) < particles.bestFitness(i)
                particles.bestFitness(i) = particles.fitness(i);
                particles.bestPositions(i,:) = particles.cartesianPositions(i,:);
                particles.stagnationCounter(i) = 0;
            else
                particles.stagnationCounter(i) = particles.stagnationCounter(i) + 1;
            end
        end

        % Track improvement
        prevBest = episodeBestFitness;
        newBest = min(particles.bestFitness);
        if newBest < episodeBestFitness
            episodeBestFitness = newBest;
            lastImprovementIter = iter;
        end
        fitnessHistory(iter) = episodeBestFitness;

        % --- Store transition and train at observation points ---
        if shouldObserve
            % Build next state
            if usePaperState
                nextState = buildPaperState(particles, iter, maxIter, config);
            else
                [nextFeatures, particles] = extractRichFeatures(particles, min(iter+1, maxIter), ...
                    maxIter, lastImprovementIter, episodeBestFitness);
                stateEncoder.addIteration(nextFeatures);
                nextState = stateEncoder.encode();
            end

            % Compute reward
            if usePaperReward
                reward = calculatePaperReward(lastObsFitness, episodeBestFitness);
                lastObsFitness = episodeBestFitness;
            else
                reward = 2 * (prevBest - episodeBestFitness) / ...
                    (abs(prevBest) + abs(episodeBestFitness) + 1e-8);
            end

            % Store and train
            done = (iter == maxIter);
            agent.storeTransition(state, action, reward, nextState, done);

            if iter > config.warmupPeriod
                for g = 1:config.gradientStepsPerTraining
                    agent.train();
                end
            end

            state = nextState;
        end

    end

    [bestFitness, bestIdx] = min(particles.bestFitness);
    bestX = particles.bestPositions(bestIdx,:);
end
