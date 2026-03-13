%% SAEPSO Parameter Control Demo
% Standalone visualization of how SAEPSO adapts parameters (w, c1, c2)
% for each particle based on fitness during optimization
%
% This demo uses a simple sphere function to clearly show parameter adaptation
% across iterations and particles.

clear; clc; close all;

%% Configuration
popSize = 10;           % Number of particles
maxIterations = 100;    % Optimization iterations
dims = 2;               % Problem dimensions

% SAEPSO parameter bounds (from config)
c_min = 0.5;
c_max = 2.5;
w_min = 0.1;
w_max = 0.9;

% Simple optimization problem: Sphere function
% f(x) = sum(x^2), optimal at x = [0, 0]
objectiveFunction = @(x) sum(x.^2, 2);

%% Initialize particles
particles = struct();
particles.positions = randn(popSize, dims) * 5;  % Random start in [-5, 5]
particles.velocities = randn(popSize, dims) * 0.1;
particles.fitness = objectiveFunction(particles.positions);
particles.bestPositions = particles.positions;
particles.bestFitness = particles.fitness;
particles.fitnessHistory = repmat(particles.fitness, 1, 3);  % For local optima detection

globalBestPosition = particles.bestPositions(1, :);
globalBestFitness = particles.bestFitness(1);

for i = 1:popSize
    if particles.fitness(i) < globalBestFitness
        globalBestFitness = particles.fitness(i);
        globalBestPosition = particles.positions(i, :);
    end
end

% Storage for parameter history
paramHistory = struct();
paramHistory.w = zeros(popSize, maxIterations);
paramHistory.c1 = zeros(popSize, maxIterations);
paramHistory.c2 = zeros(popSize, maxIterations);
paramHistory.fitness = zeros(popSize, maxIterations);
paramHistory.fap = zeros(popSize, maxIterations);

convergenceHistory = zeros(maxIterations, 1);
avgFitnessHistory = [repmat(sum(particles.fitness)/popSize, 3, 1)];

fprintf('SAEPSO Parameter Control Demo\n');
fprintf('==============================\n');
fprintf('Population: %d | Iterations: %d | Dimensions: %d\n', popSize, maxIterations, dims);
fprintf('c_min=%.2f, c_max=%.2f | w_min=%.2f, w_max=%.2f\n\n', c_min, c_max, w_min, w_max);

%% Main optimization loop
for iter = 1:maxIterations
    % Calculate average fitness
    avgPopulationFitness = sum(particles.fitness) / popSize;
    avgFitnessHistory = [avgFitnessHistory(2:end); avgPopulationFitness];

    % SAEPSO sine factor (Equation 44 from paper)
    chi = 0.5 * sin((iter/maxIterations) * pi - pi/2) + 0.5;

    % Update each particle
    for i = 1:popSize
        % Calculate adaptive fitness factor (Equation 42)
        if abs(avgPopulationFitness - globalBestFitness) > eps
            fap = (particles.fitness(i) - globalBestFitness) / (avgPopulationFitness - globalBestFitness);
        else
            fap = 0;
        end
        fap = max(0, min(1, fap));

        % SAEPSO adaptive parameters (Equations 43-44)
        w = w_max - (w_max - w_min) * (iter/maxIterations) * fap;
        c1 = c_max - (c_max - c_min) * chi * fap;
        c2 = c_min + (c_max - c_min) * chi * fap;

        % Store parameters for visualization
        paramHistory.w(i, iter) = w;
        paramHistory.c1(i, iter) = c1;
        paramHistory.c2(i, iter) = c2;
        paramHistory.fitness(i, iter) = particles.fitness(i);
        paramHistory.fap(i, iter) = fap;

        % PSO velocity update
        r1 = rand(1, dims);
        r2 = rand(1, dims);

        cognitiveComponent = c1 .* r1 .* (particles.bestPositions(i, :) - particles.positions(i, :));
        socialComponent = c2 .* r2 .* (globalBestPosition - particles.positions(i, :));

        particles.velocities(i, :) = w * particles.velocities(i, :) + cognitiveComponent + socialComponent;

        % Velocity clamping
        maxVel = 5;
        particles.velocities(i, :) = max(-maxVel, min(maxVel, particles.velocities(i, :)));

        % Position update
        particles.positions(i, :) = particles.positions(i, :) + particles.velocities(i, :);

        % Evaluate
        particles.fitness(i) = objectiveFunction(particles.positions(i, :));

        % Update personal best
        if particles.fitness(i) < particles.bestFitness(i)
            particles.bestFitness(i) = particles.fitness(i);
            particles.bestPositions(i, :) = particles.positions(i, :);
        end

        % Update global best
        if particles.fitness(i) < globalBestFitness
            globalBestFitness = particles.fitness(i);
            globalBestPosition = particles.positions(i, :);
        end
    end

    convergenceHistory(iter) = globalBestFitness;

    if mod(iter, 20) == 0
        fprintf('Iteration %3d: Global Best = %.6f, Avg Fitness = %.6f\n', ...
            iter, globalBestFitness, avgPopulationFitness);
    end
end

fprintf('\nOptimization Complete!\n');
fprintf('Final Best Fitness: %.6f\n', globalBestFitness);

%% Visualization
figure('Position', [100, 100, 1400, 900], 'Name', 'SAEPSO Parameter Control Visualization');

% Plot 1: Global convergence
ax1 = subplot(2, 3, 1);
plot(convergenceHistory, 'LineWidth', 2, 'Color', [0.2 0.4 0.8]);
xlabel('Iteration', 'FontSize', 11);
ylabel('Best Fitness', 'FontSize', 11);
title('Global Convergence', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
set(ax1, 'YScale', 'log');

% Plot 2: Inertia weight (w) for all particles
ax2 = subplot(2, 3, 2);
hold on;
colors = parula(popSize);
for i = 1:popSize
    plot(paramHistory.w(i, :), 'Color', colors(i, :), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('P%d', i));
end
xlabel('Iteration', 'FontSize', 11);
ylabel('w (Inertia Weight)', 'FontSize', 11);
title('Inertia Weight per Particle', 'FontSize', 12, 'FontWeight', 'bold');
ylim([w_min - 0.05, w_max + 0.05]);
grid on;
legend('FontSize', 8, 'Location', 'best');

% Plot 3: Cognitive coefficient (c1) for all particles
ax3 = subplot(2, 3, 3);
hold on;
for i = 1:popSize
    plot(paramHistory.c1(i, :), 'Color', colors(i, :), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('P%d', i));
end
xlabel('Iteration', 'FontSize', 11);
ylabel('c1 (Cognitive)', 'FontSize', 11);
title('Cognitive Coefficient per Particle', 'FontSize', 12, 'FontWeight', 'bold');
ylim([c_min - 0.2, c_max + 0.2]);
grid on;

% Plot 4: Social coefficient (c2) for all particles
ax4 = subplot(2, 3, 4);
hold on;
for i = 1:popSize
    plot(paramHistory.c2(i, :), 'Color', colors(i, :), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('P%d', i));
end
xlabel('Iteration', 'FontSize', 11);
ylabel('c2 (Social)', 'FontSize', 11);
title('Social Coefficient per Particle', 'FontSize', 12, 'FontWeight', 'bold');
ylim([c_min - 0.2, c_max + 0.2]);
grid on;

% Plot 5: Adaptive fitness factor (fap) for all particles
ax5 = subplot(2, 3, 5);
hold on;
for i = 1:popSize
    plot(paramHistory.fap(i, :), 'Color', colors(i, :), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('P%d', i));
end
xlabel('Iteration', 'FontSize', 11);
ylabel('f_ap (Adaptive Fitness Factor)', 'FontSize', 11);
title('Fitness Adaptive Factor per Particle', 'FontSize', 12, 'FontWeight', 'bold');
ylim([-0.05, 1.05]);
grid on;

% Plot 6: Parameter space (w vs c1, colored by iteration)
ax6 = subplot(2, 3, 6);
colorData = repmat(1:maxIterations, popSize, 1);
colorData = colorData(:);
scatter(paramHistory.w(:), paramHistory.c1(:), 30, colorData, ...
    'filled', 'MarkerFaceAlpha', 0.6);
c = colorbar;
c.Label.String = 'Iteration';
xlabel('w (Inertia Weight)', 'FontSize', 11);
ylabel('c1 (Cognitive)', 'FontSize', 11);
title('Parameter Space Trajectory', 'FontSize', 12, 'FontWeight', 'bold');
xlim([w_min - 0.05, w_max + 0.05]);
ylim([c_min - 0.2, c_max + 0.2]);
grid on;

sgtitle(sprintf('SAEPSO Parameter Control: %d Particles, %d Iterations', popSize, maxIterations), ...
    'FontSize', 14, 'FontWeight', 'bold');

%% Summary statistics
fprintf('\n=== Parameter Statistics ===\n');
fprintf('Inertia Weight (w):\n');
fprintf('  Mean: %.4f | Std: %.4f | Range: [%.4f, %.4f]\n', ...
    mean(paramHistory.w(:)), std(paramHistory.w(:)), min(paramHistory.w(:)), max(paramHistory.w(:)));

fprintf('Cognitive Coefficient (c1):\n');
fprintf('  Mean: %.4f | Std: %.4f | Range: [%.4f, %.4f]\n', ...
    mean(paramHistory.c1(:)), std(paramHistory.c1(:)), min(paramHistory.c1(:)), max(paramHistory.c1(:)));

fprintf('Social Coefficient (c2):\n');
fprintf('  Mean: %.4f | Std: %.4f | Range: [%.4f, %.4f]\n', ...
    mean(paramHistory.c2(:)), std(paramHistory.c2(:)), min(paramHistory.c2(:)), max(paramHistory.c2(:)));

fprintf('\nKey Observations:\n');
fprintf('• Parameters vary per particle based on individual fitness\n');
fprintf('• Better particles (lower fitness) get stronger exploration (higher w)\n');
fprintf('• Worse particles get more exploitation (lower c1, higher c2)\n');
fprintf('• Parameters change over iterations: early exploration → late exploitation\n');
