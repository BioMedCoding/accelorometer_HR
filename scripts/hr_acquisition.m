% Initialization
clear
close all
clc

%% Filtering parameters - initials

select_data = true;

% Initial and ending value 
starting_sample = 600;
ending_sample = 1000000;

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

save_figures = false; 
%% Section parameters
filter_data_classic = false;
%filter_data_optimised = ~filter_data_classic;
filter_data_optimised = false;
filter_data_attenuation = ~filter_data_classic;

plot_raw_data = true;

plot_tf_filters = false;

show_original_psd = false;
show_filtered_psd = false;
    plot_signal_psd_comparison = show_filtered_psd&show_filtered_psd;

plot_signal_comparison = true;

use_exponential = true;
use_cwt = false;
%% Findpeaks parameters
min_peak_prominence = 0.05; % 0.01; for breath rate
min_peak_time_distance = 0.35; % Expressed in seconds
min_threshold = 0.00001;


%% Data import and separation
%attitude = readmatrix("Signals\test\)
%attitude_t = attitude(:,1);
%attitude_phi = attitude(:,2);
%attitude_theta = attitude(:,3);
%attitude_psi = attitude(:,4);

%raw_data = readmatrix("Signals\Andràs_resting\Raw Data.csv");

% Chiedi all'utente di selezionare il file CSV
if select_data
    [filename, pathname] = uigetfile('*.csv', 'Seleziona il file CSV con i dati grezzi');
    
    % Verifica se l'utente ha selezionato un file
    if isequal(filename, 0)
        disp('Nessun file selezionato');
    else
        % Leggi i dati dal file selezionato
        filepath = fullfile(pathname, filename);
        raw_data = readmatrix(filepath);
    end

else
    raw_data = readmatrix("C:\Users\matte\Documents\GitHub\accelo_HR\data\Andràs_mediumIntensity\Raw Data.csv");
end

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

    if save_figures
        plot_name = 'raw_signal';
        save_existing_plot(filepath, gcf, plot_name);
    end
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
                
                if plot_tf_filters
                    figure
                    freqz(bH, aH, 512, fs)
                end
                % High-pass applicaiton
                filtData_x = filtfilt(bH, aH, rawData_x);
                filtData_y = filtfilt(bH, aH, rawData_y);
                filtData_z = filtfilt(bH, aH , rawData_z);
                filtData_total = filtfilt(bH, aH, rawData_total); 

                % Low-pass applicaiton
                
                if plot_tf_filters
                    figure
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

% Definizione dei parametri per il rilevamento dei picchi
min_peak_distance = fs * min_peak_time_distance;

% Altri parametri utili
% 'MinPeakHeight'
% 'Threshold': altezza minima che un picco deve avere rispetto ai vicini
[pks, locs, w, p] = findpeaks(filtData_total, 'Threshold', min_threshold, 'MinPeakDistance', min_peak_distance, 'MinPeakProminence', min_peak_prominence);

% Visualizzazione dei picchi rilevati
figure
plot(locs/fs, pks, 'O') % Convert locs to seconds
hold on
plot((1:length(filtData_total))/fs, filtData_total) % Convert samples to seconds
xlabel('Time [s]')
ylabel('Acceleration [m/s^2]')
title('HR peak detected on Original signal')
legend('Heart beat', 'Filtered signal')

% Calcolo della differenza di tempo tra i picchi (in campioni)
dt_medio = diff(locs);

% Conversione delle differenze di tempo in secondi (se fs è in Hz)
dt_medio_sec = dt_medio / fs;

% Calculate instantaneous heart rate
instantaneous_hr = 60 ./ dt_medio_sec; % Convert intervals to BPM

% Calculate the time points for each HR value (midpoint between peaks)
hr_time = (locs(1:end-1) + locs(2:end))/2 / fs;

% Plot the evolution of heart rate
figure;
plot(hr_time, instantaneous_hr);
xlabel('Time [s]');
ylabel('Heart Rate [BPM]');
title('Evolution of Heart Rate');

% Calculate the mean heart rate during the first 3 seconds
first_3_sec_indices = hr_time <= 3;
mean_hr_first_3_sec = mean(instantaneous_hr(first_3_sec_indices));
disp(['\n Mean HR during the first 3 seconds: ', num2str(mean_hr_first_3_sec), ' BPM \n']);

% Calculate the mean heart rate in the 3 seconds around the 60th second
around_60_sec_indices = hr_time >= 58.5 & hr_time <= 61.5;
mean_hr_around_60_sec = mean(instantaneous_hr(around_60_sec_indices));
disp(['\nMean HR in the 3 seconds around the 60th second: ', num2str(mean_hr_around_60_sec), ' BPM \n']);



% Numero di picchi consecutivi da utilizzare per il calcolo della HR
n = 10;

% Calcolo dell'evoluzione della frequenza cardiaca utilizzando n picchi consecutivi
num_windows = floor(length(dt_medio_sec) / (n - 1));
hr_evolution = zeros(1, num_windows);
hr_evolution_time = zeros(1, num_windows);

for i = 1:num_windows
    start_idx = (i - 1) * (n - 1) + 1;
    end_idx = start_idx + n - 2;
    if end_idx <= length(dt_medio_sec)
        hr_evolution(i) = 60 / mean(dt_medio_sec(start_idx:end_idx));
        % Calculate the midpoint time for the current window
        hr_evolution_time(i) = mean(hr_time(start_idx:end_idx));
    end
end

fprintf('\n The median HR frequency is %.1f \n', median(hr_evolution))

% Visualizzazione dell'evoluzione della frequenza cardiaca
figure;
plot(hr_evolution);
title('Heart Rate Evolution Over Time - RAW');
xlabel('Window Number');
ylabel('Heart Rate (bpm)');
grid on;

if save_figures
    plot_name = 'Heart rate evolution - RAW';
    save_existing_plot(filepath, gcf, plot_name);
end
%% Exponenttial data fitting
if use_exponential
    exp_data = hr_evolution;
    
    %% DATA CLEANING
    % Define X based on the length of cwt_data
    %X = (0:length(exp_data)-1)';  % Generating X values assuming they are uniformly spaced
    X = hr_evolution_time;
    
    % Extract Y from cwt_data
    Y = exp_data(:);  % Ensure Y is a column vector
    
    % Parameters
    windowSize = 20;
    thresholdFactorHigher = 4;
    thresholdFactorLower = 0.3;
    overlapPercentage = 0.2;  % Set the overlap percentage
    
    % Calculate the step size based on the overlap percentage
    stepSize = windowSize * (1 - overlapPercentage / 100);
    
    % Initialize arrays to store cleaned data and discarded points
    cleanedX = [];
    cleanedY = [];
    discardedX = [];
    discardedY = [];
    
    % Loop over the data with a sliding window
    i = 1;
    while i <= length(Y)
        % Define the window range
        startIdx = max(1, round(i));
        endIdx = min(length(Y), round(i + windowSize - 1));
        
        % Extract the window data
        windowData = Y(startIdx:endIdx);
        
        % Calculate mean and standard deviation of the window
        windowMean = mean(windowData);
        %windowMean = median(windowData);
        windowStd = std(windowData);
        
        % Check each point in the window
        for j = startIdx:endIdx
            if Y(j) >= (windowMean - thresholdFactorLower * windowStd) && Y(j) <= (windowMean + thresholdFactorHigher * windowStd)
                cleanedX(end+1) = X(j);
                cleanedY(end+1) = Y(j);
            else
                discardedX(end+1) = X(j);
                discardedY(end+1) = Y(j);
            end
        end
        
        % Move the window
        i = i + stepSize;
    end
    
    % Plot the original data, cleaned data, and discarded data
    fontSize = 7;
    
    figure;
    plot(X, Y, 'b*', 'LineWidth', 2, 'MarkerSize', 6);  % Original data
    hold on;
    plot(cleanedX, cleanedY, 'go', 'LineWidth', 2, 'MarkerSize', 6);  % Cleaned data
    plot(discardedX, discardedY, 'rx', 'LineWidth', 2, 'MarkerSize', 6);  % Discarded data
    grid on;
    titleString = sprintf('Data Cleaning with %.0f-Sample Window', windowSize);
    title(titleString, 'FontSize', fontSize);
    xlabel('X', 'FontSize', fontSize);
    ylabel('Y', 'FontSize', fontSize);
    legend('Original Data', 'Cleaned Data', 'Discarded Data', 'Location', 'best');
    legendHandle.FontSize = 10;
    
    if save_figures
        plot_name = 'Heart rate data cleaning';
        save_existing_plot(filepath, gcf, plot_name);
    end
    %% Exponential decay
    x = cleanedX;
    y = cleanedY;
    tbl = table(x', y');
    
    % Define the model as Y = a * exp(-b * x) + c
    modelfun = @(b,x) b(1) * exp(-b(2)*x(:, 1)) + b(3);
    
    % Initial guess for the parameters [a, b, c]
    initial_guess_a = max(y) - min(y);  % a should be roughly the range of y
    initial_guess_b = 0.1;              % b should be a small positive number
    initial_guess_c = min(y);           % c should be around the minimum of y
    beta0 = [initial_guess_a, initial_guess_b, initial_guess_c];
    
    % Fit the model
    mdl = fitnlm(tbl, modelfun, beta0);
    
    % Extract the coefficient values from the model object.
    coefficients = mdl.Coefficients{:, 'Estimate'};
    
    % Print the decay parameters to the terminal
    fprintf('Decay parameters:\n');
    fprintf('a = %.6f\n', coefficients(1));
    fprintf('b = %.6f\n', coefficients(2));
    fprintf('c = %.6f\n', coefficients(3));
    
    % Create smoothed/regressed data using the model:
    yFitted = coefficients(1) * exp(-coefficients(2)*x) + coefficients(3);
    
    % Plot the fitted model along with the original data
    figure;
    hold on;
    plot(x, y, 'b*', 'LineWidth', 2, 'MarkerSize', 5); % Noisy data
    plot(x, yFitted, 'r-', 'LineWidth', 2);             % Fitted curve
    grid on;
    title('Exponential Regression with fitnlm()', 'FontSize', fontSize);
    xlabel('Samples window', 'FontSize', fontSize);
    ylabel('BPM', 'FontSize', fontSize);
    legendHandle = legend('Noisy Y', 'Fitted Y', 'Location', 'north');
    legendHandle.FontSize = fontSize;
    formulaString = sprintf('Y = %.3f * exp(-%.3f * X) + %.3f', coefficients(1), coefficients(2), coefficients(3));
    text(mean(x), max(y), formulaString, 'FontSize', 15, 'FontWeight', 'bold');
    
    % Set up figure properties:
    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
    set(gcf, 'Name', 'Exponential Decay Fit', 'NumberTitle', 'Off');
    
    if save_figures
        plot_name = 'Heart rate evolution - Exponential fit';
        save_existing_plot(filepath, gcf, plot_name);
    end
end
%%  Envelope of the obtained data
% Doesn't work
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

%% polyfit
% Not enough descriptive for this case

% Interpolate the hr_evolution using a 2nd order polynomial
% Create a vector of indices corresponding to each hr_evolution value
% x = 1:length(hr_evolution);
% 
% % Fit a 2nd order polynomial to the hr_evolution data
% p = polyfit(x, hr_evolution, 4);
% 
% % Define a range of x values for interpolation
% x_interp = linspace(1, length(hr_evolution), 100);
% 
% % Evaluate the polynomial at the interpolated points
% hr_interp = polyval(p, x_interp);
% 
% % Plot the interpolated polynomial curve
% figure
% plot(x_interp, hr_interp, 'r-', 'LineWidth', 2, 'DisplayName', 'Interpolated HR Evolution');
% 
% % Add legend
% legend('show');
% hold off;

%% Wavelet

%Not working, maybe can be optimized but there's not enough time
if use_cwt
    cwt_signal = filtData_total;
    % Parametri del segnale
    fs = 1000; % Frequenza di campionamento in Hz
    finish_time = 0.1;
    t = 0:1/fs:finish_time; % Asse temporale di 1 secondo
    
    % Parametri del primo picco principale
    amp1 = 1; % Ampiezza del primo picco
    freq1 = 15; % Frequenza del primo picco in Hz
    duration1 = 0.033; % Durata del primo picco in secondi
    start_time1 = 0.0; % Tempo di inizio del primo picco in secondi
    
    % Parametri del secondo picco secondario
    amp2 = 0.2; % Ampiezza del secondo picco
    freq2 = 30; % Frequenza del secondo picco in Hz
    duration2 = 0.01; % Durata del secondo picco in secondi
    start_time2 = 0.038; % Tempo di inizio del secondo picco in secondi
    
    % Creazione del segnale di riferimento chiamando la funzione
    signal = create_signal(fs, t, amp1, freq1, duration1, start_time1, amp2, freq2, duration2, start_time2, finish_time);
    
    signal = signal(1:44);
    t = t(1:44);
    
    % Define the new number of points
    new_num_points = 12;
    
    % Generate new time vector with 12 points evenly spaced between the min and max of the original time vector
    t_resampled = linspace(min(t), max(t), new_num_points);
    
    % Use interp1 to resample the signal at the new time points
    signal_resampled = interp1(t, signal, t_resampled, 'linear'); % You can also use other interpolation methods like 'spline'
    
    % Plot the original and resampled signals for comparison
    figure;
    plot(t, signal, 'o-', 'DisplayName', 'Original Signal');
    hold on;
    plot(t_resampled, signal_resampled, 'x-', 'DisplayName', 'Resampled Signal');
    xlabel('Time');
    ylabel('Signal');
    title('Original and Resampled Signals');
    legend('show');
    grid on;
    
    % Perform CWT on the original signal
    [cfs, frequencies] = cwt(cwt_signal, 'amor', fs);
    
    % Plot the CWT scalogram
    figure;
    t_cwt = (0:length(cwt_signal)-1)/fs;
    imagesc(t_cwt, frequencies, abs(cfs));
    axis xy;
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    title('CWT Scalogram');
    colorbar;
    
    % Identify peaks based on a threshold
    threshold = 0.3;
    heartbeat_instants = [];
    
    for i = 1:length(t_cwt)
        if max(abs(cfs(:, i))) > threshold
            heartbeat_instants = [heartbeat_instants, t_cwt(i)];
        end
    end
    
    % Calculate heart rate in beats per minute
    heartbeat_instants_diff = diff(heartbeat_instants);
    heart_rate = 60 ./ heartbeat_instants_diff;
    
    % Display heart rate
    disp('Heart rate (BPM):');
    disp(heart_rate);
    
    % Find the closest indices in the cwt_signal
    heartbeat_indices = arrayfun(@(x) find(abs(t_cwt - x) == min(abs(t_cwt - x)), 1), heartbeat_instants);
    
    % Plot the original signal with identified peaks
    figure;
    plot(t_cwt, cwt_signal, 'b');
    hold on;
    plot(t_cwt(heartbeat_indices), cwt_signal(heartbeat_indices), 'ro');
    xlabel('Time (s)');
    ylabel('Amplitude');
    title('Original Signal with Identified Peaks');
    legend('Original Signal', 'Identified Peaks');
    grid on;
    hold off;
    
    
    
    % Plot the correlation color map
    plot_correlation_color_map(cfs, t_cwt, frequencies);
end
%% Normalize the data (optional but recommended for neural network training)

% X = cleanedX;
% Y = cleanedY;
% 
% [Xn, Xsettings] = mapminmax(X');
% [Yn, Ysettings] = mapminmax(Y');
% 
% % Create and configure the neural network
% hiddenLayerSize = 1;  % Number of neurons in the hidden layer
% net = fitnet(hiddenLayerSize);  % Create a feedforward network
% 
% % Set up division of data for training, validation, and testing
% net.divideParam.trainRatio = 70/100;  % 70% training data
% net.divideParam.valRatio = 15/100;  % 15% validation data
% net.divideParam.testRatio = 15/100;  % 15% test data
% 
% % Train the neural network
% [net, tr] = train(net, Xn, Yn);
% 
% % Test the neural network
% Y_pred = net(Xn);
% 
% % Unnormalize the predicted data
% Y_pred_unnorm = mapminmax('reverse', Y_pred, Ysettings);
% 
% % Plot the original data and the fitted curve
% figure;
% plot(X, Y, 'b*', 'LineWidth', 2, 'MarkerSize', 15);
% hold on;
% plot(X, Y_pred_unnorm, 'r-', 'LineWidth', 2);
% grid on;
% title('Neural Network Regression', 'FontSize', fontSize);
% xlabel('X', 'FontSize', fontSize);
% ylabel('Y', 'FontSize', fontSize);
% legendHandle = legend('Original Data', 'Fitted Data', 'Location', 'north');
% legendHandle.FontSize = 15;
% 
% % View the network
% %view(net);
% 
% % Display the performance
% performance = perform(net, Yn, Y_pred);
% fprintf('Performance (MSE): %.4f\n', performance);
% 
% % Set up figure properties:
% % Enlarge figure to full screen.
% set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
% % Get rid of tool bar and pulldown menus that are along top of figure.
% % set(gcf, 'Toolbar', 'none', 'Menu', 'none');
% % Give a name to the title bar.
% set(gcf, 'Name', 'Neural Network Fit with cwt_data', 'NumberTitle', 'Off');




function signal = create_signal(fs, t, amp1, freq1, duration1, start_time1, amp2, freq2, duration2, start_time2, finish_time)
    % Creazione del segnale di riferimento
    signal = zeros(size(t));
    
    % Primo picco
    start_idx1 = ceil(start_time1 * fs)+1;
    end_idx1 = start_idx1 + round(duration1 * fs) - 1;
    sig = amp1 * sin(2 * pi * freq1 * (0:1/fs:finish_time));
    signal(start_idx1:end_idx1) = signal(start_idx1:end_idx1) + sig(start_idx1:end_idx1);
    clear sig
    
    % Secondo picco
    start_idx2 = ceil(start_time2 * fs)+1;
    end_idx2 = start_idx2 + round(duration2 * fs) - 1;
    sig = amp2 * sin(2 * pi * freq2 * (0:1/fs:finish_time)+2*pi);
    signal(32:32+12) = sig(5:17);
end


function save_existing_plot(filepath, plot_handle, plot_name)
    % Extract the last folder name from the filepath
    [path, ~, ~] = fileparts(filepath);
    [path, last_folder] = fileparts(path);
    
    % Split the last folder name into two parts
    parts = split(last_folder, '_');
    if length(parts) ~= 2
        error('The last folder name should contain exactly one underscore.');
    end
    
    % Construct the new folder paths
    new_folder_1 = parts{1};
    new_folder_2 = parts{2};
    
    % Construct the new filepath for saving the plot
    new_filepath_base = fullfile(path, new_folder_1, new_folder_2);
    
    % Create directories if they do not exist
    if ~exist(new_filepath_base, 'dir')
        mkdir(new_filepath_base);
    end
    
    % File paths for PNG and FIG formats
    new_filepath_png = fullfile(new_filepath_base, [plot_name, '.png']);
    new_filepath_fig = fullfile(new_filepath_base, [plot_name, '.fig']);
    
    % Save the existing plot as a PNG file
    saveas(plot_handle, new_filepath_png);
    % Save the existing plot as a FIG file
    saveas(plot_handle, new_filepath_fig);
    
    disp(['Plot saved as PNG to: ' new_filepath_png]);
    disp(['Plot saved as FIG to: ' new_filepath_fig]);
end

% Function to plot correlation color map
function plot_correlation_color_map(cfs, t, frequencies)
    correlation_matrix = corrcoef(abs(cfs'));
    figure;
    imagesc(t, frequencies, correlation_matrix);
    axis xy;
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    title('Correlation Color Map');
    colorbar;
end
