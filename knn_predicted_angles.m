function predicted_angle = knn_predicted_angle(training_train, test_data, K)
    % Calculate distances between test spike train and training spike trains
    distances = sum((training_train - test_data).^2, [1, 2]);

    % Find indices of the K nearest neighbors
    [~, indices] = mink(distances, K);
    
    % Select angles corresponding to the K nearest neighbors
    k_nearest_labels = mod(indices - 1, size(training_train, 3)) + 1;
    
    % Select the majority angle among the K nearest neighbors
    predicted_angle = mode(k_nearest_labels);
end
