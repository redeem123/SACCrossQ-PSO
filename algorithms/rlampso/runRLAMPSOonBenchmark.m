function [bestFitness, convergenceHistory, trainedAgent] = runRLAMPSOonBenchmark(fitnessFcn, dimensions, lowerBound, upperBound, optimalValue, config, preloadedAgent)
    % Run RLAMPSO on an arbitrary benchmark function.
    %
    % Generic version of runRLAMPSOonCEC — accepts any fitness function handle.
    %
    % Inputs:
    %   fitnessFcn     — @(x) handle, x is Dx1 column vector, returns scalar
    %   dimensions     — search space dimensionality
    %   lowerBound     — scalar lower bound (applied to all dimensions)
    %   upperBound     — scalar upper bound (applied to all dimensions)
    %   optimalValue   — known global minimum (NaN to disable early-stop)
    %   config         — RLAMPSO_Config struct
    %   preloadedAgent — pre-initialized DDPG agent ([] for fresh init)
    %
    % Outputs:
    %   bestFitness        — best fitness found
    %   convergenceHistory — gbest per iteration
    %   trainedAgent       — DDPG agent after online training

    if nargin < 5 || isempty(optimalValue), optimalValue = NaN; end
    if nargin < 6 || isempty(config), config = RLAMPSO_Config('baseline'); end
    if nargin < 7, preloadedAgent = []; end

    verbose = isfield(config, 'verbose') && config.verbose;
    trainDuringDeployment = ~isfield(config, 'trainDuringDeployment') || config.trainDuringDeployment;
    if ~isfield(config, 'paramMode') || ~strcmp(config.paramMode, '5subgroup')
        warning('RLAMPSO paper mode uses fixed 5-subgroup control; forcing paramMode to 5subgroup.');
        config.paramMode = '5subgroup';
    end

    % Initialize agent
    if ~isempty(preloadedAgent)
        rlamAgent = preloadedAgent;
    else
        rlamAgent = initializeAgentAuto(config, '');
    end

    % PSO parameters
    popSize       = config.popSize;
    maxIterations = config.maxIterations;
    numSubgroups  = 5;

    % Initialize particles
    particles = struct();
    particles.cartesianPositions = lowerBound + (upperBound - lowerBound) * rand(popSize, dimensions);
    particles.velocities         = (rand(popSize, dimensions) - 0.5) * 20;
    particles.bestPositions      = particles.cartesianPositions;
    particles.bestFitness        = Inf(popSize, 1);
    particles.fitness            = Inf(popSize, 1);

    globalBestPosition = zeros(1, dimensions);
    globalBestFitness  = Inf;
    lastImprovementIteration = 0;
    convergenceHistory = [];

    % Evaluate initial population
    for i = 1:popSize
        particles.fitness(i) = fitnessFcn(particles.cartesianPositions(i,:)');
        particles.bestFitness(i) = particles.fitness(i);
        if particles.fitness(i) < globalBestFitness
            globalBestFitness  = particles.fitness(i);
            globalBestPosition = particles.cartesianPositions(i,:);
        end
    end
    convergenceHistory = [convergenceHistory; globalBestFitness];

    % Main loop
    previousReward = 0;
    previousNextState = [];

    for iter = 1:maxIterations
        prevGBF = globalBestFitness;

        % Online training
        if trainDuringDeployment && iter > 50 && ~isempty(previousNextState) && mod(iter,4)==0
            rlamAgent = updateAgentAuto(rlamAgent, previousNextState, [], previousReward);
        end

        % State (15D sin-encoding)
        rlamState = calculatePaperExactState(particles, iter, maxIterations, lastImprovementIteration);

        % Action
        [subgroupParams, rlamAgent] = getPaperExactAction(rlamAgent, rlamState, iter);

        % Update particles — modulo subgroup assignment (paper: i % 5)
        for i = 1:popSize
            sg = mod(i - 1, numSubgroups) + 1;
            w_s = subgroupParams(sg,1);
            c1_s = subgroupParams(sg,2);
            c2_s = subgroupParams(sg,3);
                r1 = rand(1, dimensions);
                r2 = rand(1, dimensions);
                particles.velocities(i,:) = w_s * particles.velocities(i,:) ...
                    + c1_s * r1 .* (particles.bestPositions(i,:) - particles.cartesianPositions(i,:)) ...
                    + c2_s * r2 .* (globalBestPosition - particles.cartesianPositions(i,:));

                maxVel = 0.2 * (upperBound - lowerBound);
                vMag = norm(particles.velocities(i,:));
                if vMag > maxVel
                    particles.velocities(i,:) = particles.velocities(i,:) * (maxVel / vMag);
                end

                particles.cartesianPositions(i,:) = particles.cartesianPositions(i,:) + particles.velocities(i,:);
                particles.cartesianPositions(i,:) = max(lowerBound, min(upperBound, particles.cartesianPositions(i,:)));

                particles.fitness(i) = fitnessFcn(particles.cartesianPositions(i,:)');

                if (particles.bestFitness(i) - particles.fitness(i)) > 1e-6
                    particles.bestFitness(i) = particles.fitness(i);
                    particles.bestPositions(i,:) = particles.cartesianPositions(i,:);
                    if (globalBestFitness - particles.fitness(i)) > 1e-6
                        globalBestFitness  = particles.fitness(i);
                        globalBestPosition = particles.cartesianPositions(i,:);
                        lastImprovementIteration = iter;
                    end
                end
        end

        % Binary reward (Eq. 13)
        if (prevGBF - globalBestFitness) > 1e-6
            reward = 1;
        else
            reward = -1;
        end

        nextState = calculatePaperExactState(particles, iter+1, maxIterations, lastImprovementIteration);
        previousReward    = reward;
        previousNextState = nextState;
        convergenceHistory = [convergenceHistory; globalBestFitness];

        if verbose && (mod(iter,50)==0 || iter==1 || iter==maxIterations)
            if isnan(optimalValue)
                fprintf('[%d/%d] Fitness: %.6e\n', iter, maxIterations, globalBestFitness);
            else
                fprintf('[%d/%d] Fitness: %.6e | Error: %.6e\n', ...
                    iter, maxIterations, globalBestFitness, abs(globalBestFitness - optimalValue));
            end
        end

        % Early stop if optimal value is known and reached
        if ~isnan(optimalValue) && abs(globalBestFitness - optimalValue) < 1e-8
            break;
        end
    end

    if trainDuringDeployment && ~isempty(previousNextState)
        rlamAgent = trainAgentAuto(rlamAgent, previousNextState, [], previousReward);
    end

    bestFitness  = globalBestFitness;
    trainedAgent = rlamAgent;
end
