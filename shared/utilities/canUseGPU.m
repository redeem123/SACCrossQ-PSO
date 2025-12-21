function hasGPU = canUseGPU()
    % Check if GPU is available and usable for computation
    %
    % Returns:
    %   hasGPU: true if GPU is available, false otherwise

    try
        % Check if Parallel Computing Toolbox is available
        if license('test', 'Distrib_Computing_Toolbox')
            % Try to get GPU device
            g = gpuDevice();
            hasGPU = g.DeviceSupported && g.AvailableMemory > 100e6;  % At least 100MB free

            if hasGPU
                % Additional check: try a simple GPU operation
                try
                    testArray = gpuArray(rand(10, 10));
                    result = testArray * testArray;
                    clear testArray result;
                    hasGPU = true;
                catch
                    hasGPU = false;
                end
            end
        else
            hasGPU = false;
        end
    catch
        % GPU not available or error occurred
        hasGPU = false;
    end
end
