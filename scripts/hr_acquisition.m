% Initialization
clear
close all
clc

%% Filtering parameters - initials

% Initial and ending value 
starting_sample = 5000;
ending_sample = 10000000;

use_cheby2 = true;

% optimzed parameters
low_freq_vec =  0.01:0.1:0.51;
high_freq_vec =  1:0.5:7;
percH_values = 1:0.1:2;
percL_values = 0.5:0.05:1;
max_ripple_values = [0.5, 1, 1.5, 2, 3];
attenuation_values = [15, 20, 30, 45];
verbose = false;

% classic paramters
high_cutoff_freq = 5;    % 0.5
low_cutoff_freq = 0.5;   % 0.01 for breath
percH = 1.2;
percL = 0.8;
rpH = 1;
rpL = 1;

attenuation = 10;
maximum_filter_order = 20;

% Define the prototype filter with desired stopband edges
prototype_filter = struct();
prototype_filter.low_stopband = 0.5;  % Example stopband edge for low-pass filter (normalized frequency)
prototype_filter.high_stopband = 4; % Example stopband edge for high-pass filter (normalized frequency)


%% Section parameters
filter_data_classic = false;
%filter_data_optimised = ~filter_data_classic;
filter_data_optimised = false;
filter_data_attenuation = ~filter_data_classic;

plot_raw_data = true;

plot_tf_filters = true;

show_original_psd = true;
show_filtered_psd = true;
    plot_signal_psd_comparison = show_filtered_psd&show_filtered_psd;

plot_signal_comparison = true;


%% Findpeaks parameters
min_peak_prominence = 0.1; % 0.01; for breath rate
min_peak_time_distance = 0.35; % Expressed in seconds
min_threshold = 0.00001;


%% Data import and separation
%attitude = readmatrix("Signals\test\)
%attitude_t = attitude(:,1);
%attitude_phi = attitude(:,2);
%attitude_theta = attitude(:,3);
%attitude_psi = attitude(:,4);

raw_data = readmatrix("Signals\Andràs_resting\Raw Data.csv");
raw_data = raw_data(starting_sample:min(ending_sample,length(raw_data)),:); % Data segmentation
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
    xlabel("Seconds [s]")
    ylabel("Acceleration [m/s^2]")
    
    subplot(4,1,2)
    plot(time_support,rawData_y)
    title("RAW acceleration - y")
    xlabel("Seconds [s]")
    ylabel("Acceleration [m/s^2]")
    
    subplot(4,1,3)
    plot(time_support,rawData_z)
    title("RAW acceleration - z")
    xlabel("Seconds [s]")
    ylabel("Acceleration [m/s^2]")

    subplot(4,1,4)
    plot(time_support,rawData_total)
    title("RAW acceleration - total")
    xlabel("Seconds [s]")
    ylabel("Acceleration [m/s^2]")
    
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
        [psd_raw(:,i), f_raw] = psd_general(raw_signals{i},"welch",fs,normalization="no");
        psd_raw(:,i) = psd_raw(:,i) / max(psd_raw(:,i)); % Normalize the PSD
        plot(f_raw, psd_raw(:,i))
        
        % Set title and legend
        title(raw_titles(i))
        xlabel("Frequency [Hz]")
        ylabel("Relative power [a.u.]")
    end
    
    % Add a general title
    sgtitle('PSD of raw signals')
    
    
    % Link axes for better comparison
    linkaxes([subplot(4,1,1) subplot(4,1,2) subplot(4,1,3) subplot(4,1,4)], 'xy')
end

clear signals show_original_psd
%% Filter data

if filter_data_optimised
    [ranked_combinations, filtData_x, filtData_y, filtData_z, filtData_total] = optimize_filter_params(rawData_x, rawData_y, rawData_z, rawData_total, fs, low_freq_vec, high_freq_vec, max_ripple_values, attenuation_values, percH_values, percL_values,prototype_filter, verbose);
end

if filter_data_attenuation

    stability = true;
    fNy = fs/2;

    while stability
        % High pass filter
        if use_cheby2
            [nH, WsH] = cheb2ord(low_cutoff_freq/fNy,(percL*low_cutoff_freq)/fNy , rpL, attenuation);
            [bH,aH] = cheby2(nH,attenuation,WsH,"high");
        else
            [nH, WsH] = buttord(low_cutoff_freq/fNy,(percL*low_cutoff_freq)/fNy , rpL, attenuation);
            [bH,aH] = butter(nH,WsH,"high");
        end

        stability_high = isstable(bH,aH);
        if nH > maximum_filter_order
            fprintf("\n nH over the maximum order \n")
        end

        % Low pass filter
        if use_cheby2
            [nL, WsL] = cheb2ord(high_cutoff_freq/fNy,(percH*high_cutoff_freq)/fNy , rpH, attenuation);
            [bL,aL] = cheby2(nL,attenuation,WsL,"low");
        else
            [nL, WsL] = buttord(high_cutoff_freq/fNy,(percH*high_cutoff_freq)/fNy , rpH, attenuation);
            [bL,aL] = butter(nL,WsL,"low");
        end
        freqz(bH, aH, 512, fs)
        stability_low = isstable(bL,aL);
        if nL > maximum_filter_order
            fprintf("\n nL over the maximum order \n")
        end


        if ~(stability_high&&stability_low) || nH > maximum_filter_order || nL>maximum_filter_order %|| attenuation >20
            stability = false;
            if  attenuation == 20
                fprintf("\n The filter is not stable \n")
            else
                fprintf("\nHighest attenuation possible with these parameters is %d \n", attenuation)
                fprintf("\nFiltering parameters are: low_edge %.1f, high_edge %.1f, rpL: %.1f, rpH %.1f, attenuation %.1 \n", percL*low_cutoff_freq, percH*high_cutoff_freq, rpL, rpH, attenuation)
                figure
                if plot_tf_filters
                    freqz(bH, aH, 512, fs)
                end
                % High-pass applicaiton
                filtData_x = filtfilt(bH, aH, rawData_x);
                filtData_y = filtfilt(bH, aH, rawData_y);
                filtData_z = filtfilt(bH, aH , rawData_z);
                filtData_total = filtfilt(bH, aH, rawData_total); 

                % Low-pass applicaiton
                figure
                if plot_tf_filters
                    freqz(bL, aL, 512, fs)
                end
                filtData_x = filtfilt(bL, aL, filtData_x);
                filtData_y = filtfilt(bL, aL, filtData_y);
                filtData_z = filtfilt(bL, aL, filtData_z);
                filtData_total = filtfilt(bL, aL, filtData_total);

            end
        else
            attenuation = attenuation + 2;
        end

    end
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
    xlabel("Seconds [s]")
    ylabel("Acceleration [m/s^2]")
    
    subplot(4,1,2)
    plot(time_support,filtData_y)
    hold on
    plot(time_support,rawData_y)
    title('Comparison of Filtered and Raw Data - Y')
    legend('Filtered Data', 'Raw Data')
    xlabel("Seconds [s]")
    ylabel("Acceleration [m/s^2]")
    
    subplot(4,1,3)
    plot(time_support,filtData_z)
    hold on
    plot(time_support,rawData_z)
    title('Comparison of Filtered and Raw Data - Z')
    legend('Filtered Data', 'Raw Data')
    xlabel("Seconds [s]")
    ylabel("Acceleration [m/s^2]")

    subplot(4,1,4)
    plot(time_support,filtData_total)
    hold on
    plot(time_support,rawData_total)
    title('Comparison of Filtered and Raw Data - Total')
    legend('Filtered Data', 'Raw Data')
    xlabel("Seconds [s]")
    ylabel("Acceleration [m/s^2]")

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
        [psd_filt(:,i), f_filt] = psd_general(filt_signals{i},"welch",fs,normalization="no");
        psd_filt(:,i) = psd_filt(:,i) / max(psd_filt(:,i)); % Normalize the PSD
        plot(f_filt, psd_filt(:,i))
        
        % Set title and legend
        title(filt_titles(i))
        xlabel("Frequency [Hz]")
        ylabel("Relative power [a.u.]")
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
        xlabel("Frequency [Hz]")
        ylabel("Relative power [a.u.]")
        
        subplot(4,1,2)
        plot(f_filt,psd_raw(:,2))
        hold on
        plot(f_filt,psd_filt(:,2))
        title('Comparison of Filtered and Raw Data PSD - Y')
        legend('RAW data', 'Filtered Data')
        xlabel("Frequency [Hz]")
        ylabel("Relative power [a.u.]")
        
        subplot(4,1,3)
        plot(f_filt,psd_raw(:,3))
        hold on
        plot(f_filt,psd_filt(:,3))
        title('Comparison of Filtered and Raw Data PSD - Z')
        legend('RAW data', 'Filtered Data')
        xlabel("Frequency [Hz]")
        ylabel("Relative power [a.u.]")
    
        subplot(4,1,4)
        plot(f_filt,psd_raw(:,4))
        hold on
        plot(f_filt,psd_filt(:,4))
        title('Comparison of Filtered and Raw Data PSD - Total')
        legend('RAW data', 'Filtered Data')
        xlabel("Frequency [Hz]")
        ylabel("Relative power [a.u.]")
    
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
% hold on
% plot(rawData_total)
xlabel("Samples")
ylabel("Acceleration [m/s^2]")
title("HR peak detected on Original signal")
%legend("Heart beat", "Filtered signal", "Original signal")
legend("Heart beat", "Filtered signal")

dt_medio = diff(locs);
mean_hr = 1/(mean(dt_medio)/fs)*60;
fprintf("\n The calculated HR is %.0f \n",mean_hr)

% Convert time differences to seconds (if fs is in Hz)
dt_medio_sec = dt_medio / fs;
% Calculate heart rate evolution over time
hr_evolution = 60 ./ dt_medio_sec;

fprintf("\n The median HR frequency is %.1f \n", median(hr_evolution))

% Plot the heart rate evolution
figure;
plot(hr_evolution);
title('Heart Rate Evolution Over Time');
xlabel('Beat Number');
ylabel('Heart Rate (bpm)');
grid on;

% Envelope of the obtained data
% En_cutoff_freq = 4;
% percH = 1.2;
% rpH = 1;
% attenuation = 20;
% [nEn, WsEn] = cheb2ord(En_cutoff_freq/fNy,(percH*En_cutoff_freq)/fNy , rpH, attenuation);
% [bEn,aEn] = cheby2(nEn,attenuation,WsEn,"low");
% stability_En = isstable(bEn,aEn);
% enData_total = filtfilt(bEn, aEn, hr_evolution);
% 
% % Plot the heart rate evolution
% figure;
% plot(enData_total);
% title('Heart Rate Evolution Over Time');
% xlabel('Beat Number');
% ylabel('Heart Rate (bpm)');
% grid on;

% polyfit
% Interpolate the hr_evolution using a 2nd order polynomial
% Create a vector of indices corresponding to each hr_evolution value
x = 1:length(hr_evolution);

% Fit a 2nd order polynomial to the hr_evolution data
p = polyfit(x, hr_evolution, 4);

% Define a range of x values for interpolation
x_interp = linspace(1, length(hr_evolution), 100);

% Evaluate the polynomial at the interpolated points
hr_interp = polyval(p, x_interp);

% Plot the interpolated polynomial curve
figure
plot(x_interp, hr_interp, 'r-', 'LineWidth', 2, 'DisplayName', 'Interpolated HR Evolution');

% Add legend
legend('show');
hold off;

%% NN Data creation

signal_label = zeros(size(rawData_total,1),1);
signal_label(locs) = 1;

