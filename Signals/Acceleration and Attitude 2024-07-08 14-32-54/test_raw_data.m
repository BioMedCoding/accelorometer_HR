% Initialization
clear
close all
clc

%% Filtering parameters
low_cutoff_freq = 0.1;
high_cutoff_freq = 3;
percH = 2;
percL = 0.5;

%% Kalman filter parameters

% Known final state for Kalman filter
final_velocity = [0; 0; 0];  
final_position = [0; 0; 0];  

% Adjustable parameters for noise covariance
process_noise_scale = 0.1;  % Scale factor for process noise
measurement_noise_scale = 0.1;  % Scale factor for measurement noise

% Incorporate final state knowledge
lambda = 0.1;  % Weighting factor for final velocity constraint
mu = 0.1;      % Weighting factor for final position constraint

% Process noise covariance
Q = eye(9) * process_noise_scale;
% Measurement noise covariance
R = eye(6) * measurement_noise_scale;
% Initial estimation covariance
P = eye(9) * 10;  % Larger initial P can help with faster convergence if the initial guess is far off


%% Section parameters
filter_data = true;
plot_raw_data = true;
show_original_psd = true;
plot_signal_comparison = true;
caluclate_simple_reconstruction = false;
calculate_Kalman_reconstruction = true;

%% Data import and separation
%raw_data = readmatrix("Raw Data.csv");
%attitude = readmatrix("Attitude.csv");
raw_data = readmatrix("Acceleration and Attitude 2024-07-08 14-32-54\test_movimento_3_assi\Raw Data.csv");
attitude = readmatrix("Acceleration and Attitude 2024-07-08 14-32-54\test_movimento_3_assi\Attitude.csv");

attitude_t = attitude(:,1);
attitude_phi = attitude(:,2);
attitude_theta = attitude(:,3);
attitude_psi = attitude(:,4);

rawData_t = raw_data(:,1);
rawData_x = raw_data(:,2);
rawData_y = raw_data(:,3);
rawData_z = raw_data(:,4);

% Calculate the total acceleration
rawData_total = sqrt(rawData_x.^2 + rawData_y.^2 + rawData_z.^2);

time_interval = diff(attitude_t);
fs = 1/mode(time_interval);

%% Raw data plotting
if plot_raw_data
    figure
    subplot(4,1,1)
    plot(rawData_x)
    
    subplot(4,1,2)
    plot(rawData_y)
    
    subplot(4,1,3)
    plot(rawData_z)

    subplot(4,1,4)
    plot(rawData_total)
    
    linkaxes([subplot(4,1,1) subplot(4,1,2) subplot(4,1,3) subplot(4,1,4)], 'xy')
end

%% Original PSD
if show_original_psd
    signals = {attitude_phi, attitude_theta, attitude_psi, []}; % Cell array of attitude signals
    raw_signals = {rawData_x, rawData_y, rawData_z, rawData_total}; % Cell array of raw data signals
    titles = ["Initial Normalised PSD of the signal - x", "Initial Normalised PSD of the signal - y", "Initial Normalised PSD of the signal - z", "Initial Normalised PSD of the signal - total"]; % Titles for subplots
    
    % Initial PSD of the attitude signal
    figure
    for i = 1:4
        subplot(4,1,i)
        
        % % Compute and normalize PSD for attitude signal
        % [psd, f] = psd_general(signals{i},"welch",fs);
        % psd = psd / max(psd); % Normalize the PSD
        % plot(f, psd)
        % hold on
        
        % Compute and normalize PSD for raw signal
        [psd_raw, f_raw] = psd_general(raw_signals{i},"welch",fs);
        psd_raw = psd_raw / max(psd_raw); % Normalize the PSD
        plot(f_raw, psd_raw)
        
        % Set title and legend
        title(titles(i))
        legend("Attitude", "Original")
    end
    title("Initial PSD ")
    % Link axes for better comparison
    linkaxes([subplot(4,1,1) subplot(4,1,2) subplot(4,1,3) subplot(4,1,4)], 'xy')
end

%% Filter data

if filter_data
    filtData_x = filter_general(rawData_x,"cheby2",fs,"fH",high_cutoff_freq,"fL", low_cutoff_freq,percH=percH,percL=percL);
    filtData_y = filter_general(rawData_y,"cheby2",fs,"fH",high_cutoff_freq,"fL", low_cutoff_freq,percH=percH,percL=percL);
    filtData_z = filter_general(rawData_z,"cheby2",fs,"fH",high_cutoff_freq,"fL", low_cutoff_freq,percH=percH,percL=percL);
    filtData_total = filter_general(rawData_total,"cheby2",fs,"fH",high_cutoff_freq,"fL", low_cutoff_freq,percH=percH,percL=percL);
end

if plot_signal_comparison
    figure
    
    subplot(4,1,1)
    plot(filtData_x)
    hold on
    plot(rawData_x)
    
    subplot(4,1,2)
    plot(filtData_y)
    hold on
    plot(rawData_y)
    
    subplot(4,1,3)
    plot(filtData_z)
    hold on
    plot(rawData_z)

    subplot(4,1,4)
    plot(filtData_total)
    hold on
    plot(rawData_total)

    title("Filtered data")
end

%% Reconstruct the 3D position from the filtered data

if caluclate_simple_reconstruction

    % Initial conditions

    initial_velocity = [0, 0, 0]; % Initial velocity in m/s for x, y, z

    initial_position = [0, 0, 0]; % Initial position in meters for x, y, z

    

    % CAZZATA MA ALMENO È RAPIDO PER TESTARE

    % !!!!!!!

    rawData_x = filtData_x;

    rawData_y = filtData_y;

    rawData_z = filtData_z;

    

    % Integrate acceleration to get velocity

    velocity_x = cumtrapz(rawData_t, rawData_x) + initial_velocity(1);

    velocity_y = cumtrapz(rawData_t, rawData_y) + initial_velocity(2);

    velocity_z = cumtrapz(rawData_t, rawData_z) + initial_velocity(3);

    

    % Integrate velocity to get position

    position_x = cumtrapz(rawData_t, velocity_x) + initial_position(1);

    position_y = cumtrapz(rawData_t, velocity_y) + initial_position(2);

    position_z = cumtrapz(rawData_t, velocity_z) + initial_position(3);

    

    % Plot the 3D trajectory

    figure;

    plot3(position_x, position_y, position_z);

    title('3D Trajectory from Raw Data');

    xlabel('Position X (m)');

    ylabel('Position Y (m)');

    zlabel('Position Z (m)');

    grid on;

    

    % Display final position and velocity

    disp(['Final Position: (', num2str(position_x(end)), ', ', num2str(position_y(end)), ', ', num2str(position_z(end)), ') meters']);

    disp(['Final Velocity: (', num2str(velocity_x(end)), ', ', num2str(velocity_y(end)), ', ', num2str(velocity_z(end)), ') m/s']);

end

%% Kalman filter reconstruction

if calculate_Kalman_reconstruction
    % Initialize variables
    dt = mean(diff(rawData_t));  % Time step
    num_samples = length(rawData_t);
    
    % State transition matrix
    F = [1 dt 0 0 0 0 0 0 0;
         0 1 0 0 0 0 0 0 0;
         0 0 1 dt 0 0 0 0 0;
         0 0 0 1 0 0 0 0 0;
         0 0 0 0 1 dt 0 0 0;
         0 0 0 0 0 1 0 0 0;
         0 0 0 0 0 0 1 0 0;
         0 0 0 0 0 0 0 1 0;
         0 0 0 0 0 0 0 0 1];
    
    % Control input matrix
    B = [0.5*dt^2 0 0;
         dt 0 0;
         0 0.5*dt^2 0;
         0 dt 0;
         0 0 0.5*dt^2;
         0 0 dt;
         0 0 0;
         0 0 0;
         0 0 0];
    
    % Observation matrix
    H = [0 1 0 0 0 0 0 0 0;
         0 0 0 1 0 0 0 0 0;
         0 0 0 0 0 1 0 0 0;
         0 0 0 0 0 0 1 0 0;
         0 0 0 0 0 0 0 1 0;
         0 0 0 0 0 0 0 0 1];
    
    % Initial state estimate
    x_hat = zeros(9, num_samples);
    x_hat(:,1) = [0; 0; 0; 0; 0; 0; attitude_phi(1); attitude_theta(1); attitude_psi(1)];  % Initial position, velocity, and angles
    
    % Kalman filter implementation
    for k = 2:num_samples
        % Prediction step
        x_hat(:,k) = F * x_hat(:,k-1) + B * [rawData_x(k-1); rawData_y(k-1); rawData_z(k-1)];
        P = F * P * F' + Q;
        
        % Measurement update
        z = [rawData_x(k); rawData_y(k); rawData_z(k); attitude_phi(k); attitude_theta(k); attitude_psi(k)];
        K = P * H' / (H * P * H' + R);
        x_hat(:,k) = x_hat(:,k) + K * (z - H * x_hat(:,k));
        P = (eye(9) - K * H) * P;
    end
    
    % Apply constraints for final state knowledge
    x_hat(2,end) = (1 - lambda) * x_hat(2,end) + lambda * final_velocity(1);
    x_hat(4,end) = (1 - lambda) * x_hat(4,end) + lambda * final_velocity(2);
    x_hat(6,end) = (1 - lambda) * x_hat(6,end) + lambda * final_velocity(3);
    
    x_hat(1,end) = (1 - mu) * x_hat(1,end) + mu * final_position(1);
    x_hat(3,end) = (1 - mu) * x_hat(3,end) + mu * final_position(2);
    x_hat(5,end) = (1 - mu) * x_hat(5,end) + mu * final_position(3);
    
    % Extract position estimates
    position_x_kalman = x_hat(1,:);
    position_y_kalman = x_hat(3,:);
    position_z_kalman = x_hat(5,:);
    
    % Plot the 3D trajectory with time as color
    figure;
    scatter3(position_x_kalman, position_y_kalman, position_z_kalman, 20, rawData_t, 'filled');
    title('3D Trajectory from Kalman Filtered Data');
    xlabel('Position X (m)');
    ylabel('Position Y (m)');
    zlabel('Position Z (m)');
    colormap('jet');
    colorbar;
    grid on;
    
    % Display final position and velocity
    disp(['Final Position (Kalman): (', num2str(position_x_kalman(end)), ', ', num2str(position_y_kalman(end)), ', ', num2str(position_z_kalman(end)), ') meters']);
    disp(['Final Velocity (Kalman): (', num2str(x_hat(2,end)), ', ', num2str(x_hat(4,end)), ', ', num2str(x_hat(6,end)), ') m/s']);
end
