function [x, y] = positionEstimator(test_data, modelParameters)
    bin_size = 20; % Bin size
    start_point = 1; % Start point
    end_point = 320; % End point
    num_prev_bins = 1; % Including data from the previous bin

    % Calculate the total number of bins
    num_bins = ceil((end_point - start_point) / bin_size);

    % Initialize the matrix to store binned data
    binned_test_data = zeros(size(test_data.spikes, 1), num_bins + num_prev_bins - 1); % Extend to accommodate previous bins

    % Loop over bins to calculate means with temporal context
    for bin_idx = 1:num_bins
        start_idx = (bin_idx - 1) * bin_size + start_point;
        end_idx = min(bin_idx * bin_size + start_point - 1, size(test_data.spikes, 2)); % Ensure not exceeding bounds
        
        % Calculate the mean for the current bin
        if bin_idx == 1
            % For the first bin, there's no previous bin to include
            binned_test_data(:, bin_idx) = mean(test_data.spikes(:, start_idx:end_idx), 2);
        else
            % For subsequent bins, include data from the current and previous bins
            for prev_bin_idx = max(1, bin_idx - num_prev_bins):bin_idx
                temp_start_idx = (prev_bin_idx - 1) * bin_size + start_point;
                temp_end_idx = min(prev_bin_idx * bin_size + start_point - 1, size(test_data.spikes, 2));
                % Summing instead of averaging across bins to maintain the original data structure size
                binned_test_data(:, bin_idx) = binned_test_data(:, bin_idx) + mean(test_data.spikes(:, temp_start_idx:temp_end_idx), 2);
            end
            % Average the summed data for the current and previous bins
            binned_test_data(:, bin_idx) = binned_test_data(:, bin_idx) / (num_prev_bins + (bin_idx > num_prev_bins));
        end
    end

    % Trim the extended part used for including previous bins' data
    binned_test_data = binned_test_data(:, 1:num_bins);

    % Initialize training_train variable
    training_train = zeros(98, 320/bin_size, 8);

    % Loop over angles
    for j = 1:8
        % Extract firing rates for angle j
        firing_rates = modelParameters.param(j).firing_rates(:, 1:320/bin_size);
        
        % Store firing rates in training_train variable
        training_train(:, :, j) = firing_rates;
    end
    
    % Predict angle using kNN classifier
    angle = knn_predicted_angles(training_train, binned_test_data, 1);
    
    % Retrieve parameters for predicted angle
    param = modelParameters.param(angle);
    dt = modelParameters.bin_size;
    size_min = param.size_min;
    
    % Retrieve dynamics data for predicted angle
    X = param.dynamics;

    current_time = floor(size(test_data.spikes, 2) / dt);
    
    if current_time >= size(X, 2)
        % If current time exceeds available data, use the last available position
        x = X(1, end);
        y = X(3, end);
    else
        if ~isempty(test_data.decodedHandPos)
            % If decoded hand position data is available, use it to determine position
            x = X(1, current_time);
            y = X(3, current_time);
        else
            % Otherwise, estimate position based on start hand position and dynamics
            x = test_data.startHandPos(1) + dt * X(2, 1);
            y = test_data.startHandPos(2) + dt * X(4, 1);
        end
    end
end
   