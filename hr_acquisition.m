% Initialization
clear
close all
clc

%% Filtering parameters - initials

% Initial and ending value 
starting_sample = 1000;
ending_sample = 4000;

% optimzed parameters
low_freq_vec = 0.01:0.01:0.51;
high_freq_vec = 1:0.5:7;
percH_values = 1:0.1:2;
percL_values = 0.5:0.05:1;
max_ripple = 2;
attenuation = 20;

% classic paramters
high_cutoff_freq = 3;   %0.5
low_cutoff_freq = 0.5;   %0.01 for breath
percH = 1.2;
percL = 0.5;
rpH = 2;
rpL = 2;

% Define the prototype filter with desired stopband edges
prototype_filter = struct();
prototype_filter.low_stopband = 0.5;  % Example stopband edge for low-pass filter (normalized frequency)
prototype_filter.high_stopband = 3; % Example stopband edge for high-pass filter (normalized frequency)


%% Section parameters
filter_data_classic = true;
filter_data_optimised = ~filter_data_classic;

plot_raw_data = true;

show_original_psd = true;
show_filtered_psd = true;
    plot_signal_psd_comparison = show_filtered_psd&show_filtered_psd;

plot_signal_comparison = true;


%% Findpeaks parameters
min_peak_prominence = 0.1; % 0.01; for breath rate
min_peak_time_distance = 0.25; % Expressed in seconds
min_threshold = 0.00001;


%% Data import and separation
%raw_data = readmatrix("Raw Data.csv");
%attitude = readmatrix("Attitude.csv");
%raw_data = readmatrix("Acceleration and Attitude 2024-07-08 14-32-54\test_movimento_3_assi\Raw Data.csv");
%attitude = readmatrix("Acceleration and Attitude 2024-07-08 14-32-54\test_movimento_3_assi\Attitude.csv");
raw_data = readmatrix("Raw Data_resting_Matteo.csv");
%raw_data = readmatrix("test\Linear Acceleration.csv");

raw_data = raw_data(starting_sample:min(ending_sample,length(raw_data)),:);

%attitude_t = attitude(:,1);
%attitude_phi = attitude(:,2);
%attitude_theta = attitude(:,3);
%attitude_psi = attitude(:,4);

rawData_t = raw_data(:,1);
rawData_x = raw_data(:,2);
rawData_y = raw_data(:,3);
rawData_z = raw_data(:,4);

% Calculate the total acceleration
rawData_total = sqrt(rawData_x.^2 + rawData_y.^2 + rawData_z.^2);

% Sampling frequency calculation
time_interval = diff(rawData_t);
fs = ceil(1/mode(time_interval));

% Create a variable used as graphs time support
time_support = rawData_t;

clear starting_sample ending_sample

%% Raw data plotting
if plot_raw_data
    figure
    subplot(4,1,1)
    plot(time_support,rawData_x)
    title("RAW acceleration - x")
    
    subplot(4,1,2)
    plot(time_support,rawData_y)
    title("RAW acceleration - y")
    
    subplot(4,1,3)
    plot(time_support,rawData_z)
    title("RAW acceleration - z")

    subplot(4,1,4)
    plot(time_support,rawData_total)
    title("RAW acceleration - total")
    
    linkaxes([subplot(4,1,1) subplot(4,1,2) subplot(4,1,3) subplot(4,1,4)], 'xy')

    sgtitle("RAW data")
end

clear plot_raw_data
%% Original PSD
if show_original_psd
    % signals = {attitude_phi, attitude_theta, attitude_psi}; % Cell array of attitude signals
    raw_signals = {rawData_x, rawData_y, rawData_z, rawData_total}; % Cell array of raw data signals
    raw_titles = ["Initial Normalised PSD of the signal - x", "Initial Normalised PSD of the signal - y", "Initial Normalised PSD of the signal - z", "Initial Normalised PSD of the signal - total"]; % Titles for subplots
    
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
        [psd_raw(:,i), f_raw] = psd_general(raw_signals{i},"welch",fs);
        psd_raw(:,i) = psd_raw(:,i) / max(psd_raw(:,i)); % Normalize the PSD
        plot(f_raw, psd_raw(:,i))
        
        % Set title and legend
        title(raw_titles(i))
    end
    
    % Add a general title
    sgtitle('PSD of raw signals')
    
    % Link axes for better comparison
    linkaxes([subplot(4,1,1) subplot(4,1,2) subplot(4,1,3) subplot(4,1,4)], 'xy')
end

clear signals, show_original_psd
%% Filter data

if filter_data_optimised
    [ranked_combinations, filtData_x, filtData_y, filtData_z, filtData_total] = optimize_filter_params(rawData_x, rawData_y, rawData_z, rawData_total, fs, low_freq_vec, high_freq_vec, max_ripple, percH_values, percL_values,attenuation,prototype_filter);
end

if filter_data_classic
    filtData_x = filter_general(rawData_x,"cheby2",fs,"fH",high_cutoff_freq,"fL", low_cutoff_freq,percH=percH,percL=percL,RpH=rpH,RpL=rpL);
    filtData_y = filter_general(rawData_y,"cheby2",fs,"fH",high_cutoff_freq,"fL", low_cutoff_freq,percH=percH,percL=percL,RpH=rpH,RpL=rpL);
    filtData_z = filter_general(rawData_z,"cheby2",fs,"fH",high_cutoff_freq,"fL", low_cutoff_freq,percH=percH,percL=percL,RpH=rpH,RpL=rpL);
    filtData_total = filter_general(rawData_total,"cheby2",fs,"fH",high_cutoff_freq,"fL", low_cutoff_freq,percH=percH,percL=percL,RpH=rpH,RpL=rpL);
end

if plot_signal_comparison
    figure
    
    subplot(4,1,1)
    plot(time_support,filtData_x)
    hold on
    plot(time_support,rawData_x)
    title('Comparison of Filtered and Raw Data - X')
    legend('Filtered Data', 'Raw Data')
    
    subplot(4,1,2)
    plot(time_support,filtData_y)
    hold on
    plot(time_support,rawData_y)
    title('Comparison of Filtered and Raw Data - Y')
    legend('Filtered Data', 'Raw Data')
    
    subplot(4,1,3)
    plot(time_support,filtData_z)
    hold on
    plot(time_support,rawData_z)
    title('Comparison of Filtered and Raw Data - Z')
    legend('Filtered Data', 'Raw Data')

    subplot(4,1,4)
    plot(time_support,filtData_total)
    hold on
    plot(time_support,rawData_total)
    title('Comparison of Filtered and Raw Data - Total')
    legend('Filtered Data', 'Raw Data')

    linkaxes([subplot(4,1,1) subplot(4,1,2) subplot(4,1,3) subplot(4,1,4)], 'xy')

    % Add a general title for the entire figure
    sgtitle('Filtered vs. Raw Data Comparison')
end

clear filter_data_optimised filter_data_classic plot_signal_comparison
clear attenuation high_cutoff_freq high_freq_vec low_cutoff_freq low_freq_vec percH percH_values percL percL_values prototype_filter rpH rpL 

%% Filtered PSD
if show_filtered_psd
    
    filt_signals = {filtData_x, filtData_y, filtData_z, filtData_total}; % Cell array of raw data signals
    filt_titles = ["Filtered Normalised PSD of the signal - x", "Filtered Normalised PSD of the signal - y", "Filtered Normalised PSD of the signal - z", "Filtered Normalised PSD of the signal - total"]; % Titles for subplots
    
    %psd_filt = zeros(size(raw_data));
    % Initial PSD of the attitude signal
    figure
    for i = 1:4
        subplot(4,1,i)
        
        % Compute and normalize PSD for raw signal
        [psd_filt(:,i), f_filt] = psd_general(filt_signals{i},"welch",fs);
        psd_filt(:,i) = psd_filt(:,i) / max(psd_filt(:,i)); % Normalize the PSD
        plot(f_filt, psd_filt(:,i))
        
        % Set title and legend
        title(filt_titles(i))
    end
    
    % Add a general title
    sgtitle('PSD of filtered Signals')
    
    % Link axes for better comparison
    linkaxes([subplot(4,1,1) subplot(4,1,2) subplot(4,1,3) subplot(4,1,4)], 'xy')

clear filt_signals filt_titles show_filtered_psd
    %% Comparison between PSD
    if plot_signal_psd_comparison
        figure
        
        subplot(4,1,1)
        plot(f_filt,psd_raw(:,1))
        hold on
        plot(f_filt,psd_filt(:,1))
        title('Comparison of Filtered and Raw Data PSD - X')
        legend('RAW data', 'Filtered Data')
        
        subplot(4,1,2)
        plot(f_filt,psd_raw(:,2))
        hold on
        plot(f_filt,psd_filt(:,2))
        title('Comparison of Filtered and Raw Data PSD - Y')
        legend('RAW data', 'Filtered Data')
        
        subplot(4,1,3)
        plot(f_filt,psd_raw(:,3))
        hold on
        plot(f_filt,psd_filt(:,3))
        title('Comparison of Filtered and Raw Data PSD - Z')
        legend('RAW data', 'Filtered Data')
    
        subplot(4,1,4)
        plot(f_filt,psd_raw(:,4))
        hold on
        plot(f_filt,psd_filt(:,4))
        title('Comparison of Filtered and Raw Data PSD - Total')
        legend('RAW data', 'Filtered Data')
    
        linkaxes([subplot(4,1,1) subplot(4,1,2) subplot(4,1,3) subplot(4,1,4)], 'xy')
    
        % Add a general title for the entire figure
        sgtitle('Filtered vs. Raw Data Comparison PSD')
    end

clear plot_signal_psd_comparison
end

%% Peak search
min_peak_distance = fs*min_peak_time_distance;

% Other useful parameters could be
% 'MinPeakHeight'
% 'Threshold': minimum height that must have from neighbour
[pks,locs,w,p] = findpeaks(filtData_total,'Threshold',min_threshold, 'MinPeakDistance',min_peak_distance,"MinPeakProminence",min_peak_prominence);
figure
plot(locs,pks,'O')
hold on
plot(filtData_total)
xlabel("Samples")
title("HR peak detected on Original signal")

dt_medio = diff(locs);
mean_hr = 1/(mean(dt_medio)/fs)*60;
fprintf("The calculated HR is %.0f \n",mean_hr)
