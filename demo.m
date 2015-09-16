clear all;
close all;

%% load data set
data = iris_dataset';
data_class = [ones(50, 1); 2 * ones(50, 1); 3 * ones(50, 1)];
[ninst, dim] = size(data);

%% selecting and projecting samples
sample_size = ceil(sqrt(ninst));
sample = randperm(ninst, sample_size);

%% force projection (sample projection)
options = struct();  % use default values
ys = force(data(sample, :), options);

%% kelp projection
y = kelp(data, data(sample, :), ys, options);

%% plot projection
scatter(y(:, 1), y(:, 2), 30, data_class, 'filled', 'markeredgecolor', 'k');
