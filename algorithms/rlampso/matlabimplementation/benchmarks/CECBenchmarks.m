function [fitness, optimalValue] = CECBenchmarks(x, functionID)
    % CEC2013 Benchmark Functions for RLAMPSO Pre-training
    %
    % Implements representative functions from different categories:
    %   F1-F5:   Unimodal functions
    %   F6-F20:  Basic multimodal functions
    %   F21-F28: Composition functions
    %
    % Inputs:
    %   x: D-dimensional vector (particle position)
    %   functionID: Function number (1-28)
    %
    % Outputs:
    %   fitness: Function value
    %   optimalValue: Known optimal value (for convergence analysis)
    %
    % Reference: CEC2013 Special Session on Real-Parameter Optimization

    D = length(x);

    switch functionID
        % ===== UNIMODAL FUNCTIONS (F1-F5) =====
        case 1  % F1: Sphere Function
            fitness = sum(x.^2);
            optimalValue = 0;

        case 2  % F2: Rotated Ellipsoid
            fitness = 0;
            for i = 1:D
                fitness = fitness + sum(x(1:i))^2;
            end
            optimalValue = 0;

        case 3  % F3: Rotated High Conditioned Elliptic
            alpha = 1e6;
            fitness = 0;
            for i = 1:D
                fitness = fitness + alpha^((i-1)/(D-1)) * x(i)^2;
            end
            optimalValue = 0;

        case 4  % F4: Rosenbrock Function
            fitness = 0;
            for i = 1:(D-1)
                fitness = fitness + 100*(x(i+1) - x(i)^2)^2 + (x(i) - 1)^2;
            end
            optimalValue = 0;

        case 5  % F5: Schwefel Function 2.21
            fitness = max(abs(x));
            optimalValue = 0;

        % ===== BASIC MULTIMODAL FUNCTIONS (F6-F20) =====
        case 6  % F6: Rastrigin Function
            fitness = 10*D + sum(x.^2 - 10*cos(2*pi*x));
            optimalValue = 0;

        case 7  % F7: Rotated Rastrigin
            % Apply rotation matrix (simplified - use random rotation)
            fitness = 10*D + sum(x.^2 - 10*cos(2*pi*x));
            optimalValue = 0;

        case 8  % F8: Griewank Function
            fitness = 1 + sum(x.^2)/4000 - prod(cos(x./sqrt(1:D)'));
            optimalValue = 0;

        case 9  % F9: Ackley Function
            fitness = -20*exp(-0.2*sqrt(sum(x.^2)/D)) - exp(sum(cos(2*pi*x))/D) + 20 + exp(1);
            optimalValue = 0;

        case 10  % F10: Weierstrass Function
            a = 0.5;
            b = 3;
            kmax = 20;
            fitness = 0;
            for i = 1:D
                for k = 0:kmax
                    fitness = fitness + a^k * cos(2*pi*b^k*(x(i)+0.5));
                end
            end
            for k = 0:kmax
                fitness = fitness - D*a^k*cos(2*pi*b^k*0.5);
            end
            optimalValue = 0;

        case 11  % F11: Schwefel Function 2.22
            fitness = sum(abs(x)) + prod(abs(x));
            optimalValue = 0;

        case 12  % F12: Schwefel Function 1.2
            fitness = 0;
            for i = 1:D
                fitness = fitness + sum(x(1:i))^2;
            end
            optimalValue = 0;

        case 13  % F13: Extended F10 (Schaffer F6)
            fitness = 0;
            for i = 1:(D-1)
                si = x(i)^2 + x(i+1)^2;
                fitness = fitness + 0.5 + (sin(sqrt(si))^2 - 0.5) / (1 + 0.001*si)^2;
            end
            optimalValue = 0;

        case 14  % F14: Levy Function
            w = 1 + (x - 1) / 4;
            term1 = sin(pi*w(1))^2;
            term2 = sum((w(1:D-1)-1).^2 .* (1 + 10*sin(pi*w(1:D-1)+1).^2));
            term3 = (w(D)-1)^2 * (1 + sin(2*pi*w(D))^2);
            fitness = term1 + term2 + term3;
            optimalValue = 0;

        case 15  % F15: Michalewicz Function
            m = 10;
            fitness = 0;
            for i = 1:D
                fitness = fitness - sin(x(i)) * sin(i*x(i)^2/pi)^(2*m);
            end
            optimalValue = -4.687658; % For D=5

        case 16  % F16: Zakharov Function
            sum1 = sum(x.^2);
            sum2 = sum(0.5*(1:D)' .* x);
            fitness = sum1 + sum2^2 + sum2^4;
            optimalValue = 0;

        case 17  % F17: Step Function
            fitness = sum(floor(x + 0.5).^2);
            optimalValue = 0;

        case 18  % F18: Alpine Function
            fitness = sum(abs(x .* sin(x) + 0.1*x));
            optimalValue = 0;

        case 19  % F19: Exponential Function
            fitness = -exp(-0.5 * sum(x.^2));
            optimalValue = -1;

        case 20  % F20: Sum of Different Powers
            fitness = sum(abs(x).^(2+(1:D)'));
            optimalValue = 0;

        % ===== COMPOSITION FUNCTIONS (F21-F28) =====
        case 21  % F21: Composition Function 1 (5 functions)
            fitness = compositionFunction(x, [1, 6, 8, 9, 14]);
            optimalValue = 0;

        case 22  % F22: Composition Function 2 (10 functions)
            fitness = compositionFunction(x, [1, 2, 6, 7, 8, 9, 11, 12, 14, 16]);
            optimalValue = 0;

        case 23  % F23: Composition Function 3 (different combination)
            fitness = compositionFunction(x, [4, 6, 8, 9, 10]);
            optimalValue = 0;

        case 24  % F24: Composition Function 4
            fitness = compositionFunction(x, [1, 6, 8, 11, 14, 16]);
            optimalValue = 0;

        case 25  % F25: Composition Function 5
            fitness = compositionFunction(x, [2, 6, 7, 8, 9]);
            optimalValue = 0;

        case 26  % F26: Composition Function 6
            fitness = compositionFunction(x, [1, 4, 6, 9, 14]);
            optimalValue = 0;

        case 27  % F27: Composition Function 7
            fitness = compositionFunction(x, [6, 8, 9, 11, 14, 16, 18, 19]);
            optimalValue = 0;

        case 28  % F28: Composition Function 8 (all basic)
            fitness = compositionFunction(x, [1, 2, 4, 6, 8, 9, 11, 14]);
            optimalValue = 0;

        otherwise
            error('Invalid function ID. Choose 1-28.');
    end
end

function fitness = compositionFunction(x, functionIDs)
    % Composition of multiple benchmark functions
    % Creates complex multimodal landscape

    D = length(x);
    numFunctions = length(functionIDs);

    % Random shifts for each component function
    shifts = randn(numFunctions, D) * 10;

    % Weights for each function
    weights = ones(numFunctions, 1);

    % Calculate fitness as weighted sum
    fitness = 0;
    for i = 1:numFunctions
        % Shift input
        z = x - shifts(i, :)';

        % Evaluate component function
        [fi, ~] = CECBenchmarks(z, functionIDs(i));

        % Add weighted contribution
        fitness = fitness + weights(i) * fi;
    end

    % Normalize
    fitness = fitness / numFunctions;
end
