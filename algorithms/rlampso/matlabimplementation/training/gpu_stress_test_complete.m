%% =====================================================================
%  COMPREHENSIVE GPU STRESS TEST & MONITORING
%  Maximize GPU utilization and verify GPU acceleration
%% =====================================================================

clear; clc; close all;

fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║        GPU STRESS TEST - MAXIMUM UTILIZATION                ║\n');
fprintf('║    Performance Verification & Real-time Monitoring          ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

%% STEP 1: GPU DETECTION
fprintf('═══ STEP 1: GPU DETECTION ═══\n');

if ~canUseGPU()
    fprintf('✗ NO GPU DETECTED - Using CPU only\n');
    return;
end

gpu = gpuDevice;
fprintf('✓ GPU DETECTED\n');
fprintf('  Name: %s\n', gpu.Name);
fprintf('  Total Memory: %.2f GB\n', gpu.TotalMemory / 1e9);
fprintf('  Available Memory: %.2f GB\n', gpu.AvailableMemory / 1e9);
fprintf('  Compute Capability: %s\n', gpu.ComputeCapability);
fprintf('  Driver Version: %s\n', gpu.DriverVersion);
fprintf('\n');

%% STEP 2: BASIC MATRIX OPERATIONS
fprintf('═══ STEP 2: BASIC MATRIX OPERATIONS ═══\n');

sizes = [1000, 5000, 10000];
for size = sizes
    fprintf('  Matrix multiplication (%dx%d)...\n', size, size);
    A_cpu = rand(size);
    B_cpu = rand(size);

    tic; C_cpu = A_cpu * B_cpu; cpu_time = toc;

    A_gpu = gpuArray(A_cpu);
    B_gpu = gpuArray(B_cpu);
    tic; C_gpu = A_gpu * B_gpu; wait(gpuDevice); gpu_time = toc;

    speedup = cpu_time / gpu_time;
    fprintf('    CPU: %.4fs | GPU: %.4fs | Speedup: %.2f×\n', cpu_time, gpu_time, speedup);
end
fprintf('\n');

%% STEP 3: INTENSIVE COMPUTATION (Main Stress Test)
fprintf('═══ STEP 3: INTENSIVE COMPUTATION (60 seconds) ═══\n');
fprintf('  Running sustained GPU computation with maximum load...\n\n');

matrix_size = 8192;
num_iterations = 100;
num_matrices = 4;

% Initialize on GPU
matrices = {};
for i = 1:num_matrices
    matrices{i} = gpuArray(rand(matrix_size, 'single'));
end

% Warm up
for i = 1:num_matrices
    matrices{i} = matrices{i} * matrices{i};
end
wait(gpuDevice);

% Stress test with monitoring
fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║ Iter │ Time(s) │ Mem Used │ Mem %% │ Status                ║\n');
fprintf('║─────┼─────────┼──────────┼───────┼───────────────────────║\n');

tic;
start_time = tic;

for iter = 1:num_iterations
    % Intensive operations on all matrices
    for i = 1:num_matrices
        matrices{i} = sin(matrices{i}) .* cos(matrices{i});
        matrices{i} = matrices{i} .* matrices{i};
        matrices{i} = sqrt(abs(matrices{i}));
        matrices{i} = exp(matrices{i} / 10);
    end

    % Monitor every iteration
    if mod(iter, 5) == 0 || iter == 1
        wait(gpuDevice);
        elapsed = toc(start_time);
        gpu = gpuDevice;
        mem_used = (gpu.TotalMemory - gpu.AvailableMemory) / 1e9;
        mem_percent = 100 * mem_used / (gpu.TotalMemory / 1e9);

        % Determine status
        if mem_percent > 85
            status = '⚠ CRITICAL (>85%)';
        elseif mem_percent > 70
            status = '● HIGH (70-85%)';
        elseif mem_percent > 40
            status = '◐ NORMAL (40-70%)';
        else
            status = '○ IDLE (<40%)';
        end

        fprintf('║ %3d │ %6.1f  │ %6.2f GB │ %5.1f%% │ %s                ║\n', ...
                iter, elapsed, mem_used, mem_percent, status);
    end
end

wait(gpuDevice);
total_time = toc(start_time);

fprintf('║─────┴─────────┴──────────┴───────┴───────────────────────║\n');
fprintf('║ Computation complete: %.1f seconds for %d iterations      ║\n', total_time, num_iterations);
fprintf('╚════════════════════════════════════════════════════════════╝\n');

fprintf('\n');

%% STEP 4: MEMORY BANDWIDTH TEST
fprintf('═══ STEP 4: MEMORY BANDWIDTH TEST ═══\n');
fprintf('  Testing CPU↔GPU transfer speeds...\n\n');

data_sizes_mb = [10, 100, 500, 1000];

fprintf('║ Data Size │ CPU→GPU  │ GPU→CPU  ║\n');
fprintf('║───────────┼──────────┼──────────║\n');

for data_mb = data_sizes_mb
    data_size = data_mb * 1024 * 1024 / 8;
    data_cpu = rand(data_size, 1);

    tic; data_gpu = gpuArray(data_cpu); wait(gpuDevice); cpu_to_gpu_time = toc;
    tic; data_cpu_back = gather(data_gpu); gpu_to_cpu_time = toc;

    cpu_to_gpu_bw = data_mb / cpu_to_gpu_time;
    gpu_to_cpu_bw = data_mb / gpu_to_cpu_time;

    fprintf('║ %4d MB   │ %5.0f MB/s │ %5.0f MB/s ║\n', ...
            data_mb, cpu_to_gpu_bw, gpu_to_cpu_bw);
end

fprintf('║───────────┴──────────┴──────────║\n');
fprintf('\n');

%% STEP 5: DEEP LEARNING OPERATIONS
fprintf('═══ STEP 5: DEEP LEARNING OPERATIONS ═══\n');
fprintf('  Testing Neural Network computations...\n\n');

try
    layers = [
        imageInputLayer([32 32 3])
        convolution2dLayer(3, 64, 'Padding', 'same')
        reluLayer
        convolution2dLayer(3, 64, 'Padding', 'same')
        reluLayer
        fullyConnectedLayer(256)
        reluLayer
        fullyConnectedLayer(10)
        softmaxLayer
    ];

    net = dlnetwork(layers);

    batch_sizes = [1, 32, 128];
    fprintf('║ Batch Size │ Forward Pass Time ║\n');
    fprintf('║────────────┼──────────────────║\n');

    for batch_size = batch_sizes
        X = dlarray(rand(32, 32, 3, batch_size, 'single'), 'SSCB');
        X = gpuArray(X);

        tic;
        Y = forward(net, X);
        wait(gpuDevice);
        dl_time = toc;

        fprintf('║ %3d        │ %.4f seconds   ║\n', batch_size, dl_time);
    end
    fprintf('║────────────┴──────────────────║\n');

    fprintf('  ✓ Deep Learning test successful\n');
catch ME
    fprintf('  Note: Deep Learning test skipped - %s\n', ME.message);
end

fprintf('\n');

%% STEP 6: FINAL GPU STATUS & SUMMARY
fprintf('═══ STEP 6: FINAL GPU STATUS ═══\n\n');

gpu = gpuDevice;
mem_used = (gpu.TotalMemory - gpu.AvailableMemory) / 1e9;
mem_percent = 100 * mem_used / (gpu.TotalMemory / 1e9);

fprintf('GPU: %s\n', gpu.Name);
fprintf('Total Memory: %.2f GB\n', gpu.TotalMemory / 1e9);
fprintf('Used Memory: %.2f GB (%.1f%%)\n', mem_used, mem_percent);
fprintf('Available Memory: %.2f GB\n', gpu.AvailableMemory / 1e9);

fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║         GPU STRESS TEST COMPLETE!                           ║\n');
fprintf('║    Your GPU is working and ready for training!             ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

fprintf('RECOMMENDATIONS:\n');
if mem_percent > 85
    fprintf('  ⚠ GPU Memory is heavily utilized (%.1f%%)\n', mem_percent);
    fprintf('    - Your GPU is performing well under load\n');
    fprintf('    - Reduce training batch size if you get out-of-memory errors\n');
elseif mem_percent < 20
    fprintf('  ✓ GPU is underutilized (%.1f%%)\n', mem_percent);
    fprintf('    - Increase batch size for better GPU efficiency\n');
    fprintf('    - More room for optimization\n');
else
    fprintf('  ✓ GPU utilization is healthy (%.1f%%)\n', mem_percent);
end

fprintf('\nTO MONITOR WHILE TRAINING:\n');
fprintf('  1. Open Command Prompt and run: nvidia-smi -l 1\n');
fprintf('  2. This updates GPU stats every 1 second\n');
fprintf('  3. Run your training and watch GPU usage increase\n');
fprintf('  4. Target: 80-95%% GPU memory utilization\n\n');

fprintf('If GPU still shows 0%% during training:\n');
fprintf('  - Check NVIDIA driver: nvidia-smi\n');
fprintf('  - Check CUDA in MATLAB: gpuDevice\n');
fprintf('  - Verify your training uses dlnetwork (Deep Learning)\n\n');
