clear 
close all 
clc

%% Test

% Parameters
fs = 1000; % Sampling frequency in Hz
finish_time = 0.1;
t = 0:1/fs:finish_time; % Time axis 
% First peak parameters
amp1 = 1; % Amplitude of the first peak
freq1 = 15; % Frequency of the first peak in Hz
duration1 = 0.033; % Duration of the first peak in seconds
start_time1 = 0.0; % Start time of the first peak in seconds

% Second peak parameters
amp2 = 0.2; % Amplitude of the second peak
freq2 = 30; % Frequency of the second peak in Hz
duration2 = 0.01; % Duration of the second peak in seconds
start_time2 = 0.038; % Start time of the second peak in seconds

% Create reference signal
signal = create_signal(fs, t, amp1, freq1, duration1, start_time1, amp2, freq2, duration2, start_time2, finish_time);

% Limit the signal and time vector for plotting
signal = signal(1:44);
t_signal = t(1:44);

% Plot the reference signal
figure;
plot(t_signal, signal);
title('Segnale di Riferimento con Due Picchi Sinusoidali');
xlabel('Tempo (s)');
ylabel('Ampiezza');
grid on;

% Other signals for noisy signal creation
zeros_sig = zeros(1,20);
ones_sig = ones(1,30);

% Sinusoid creation
t_noisy = 0:1/fs:1-1/fs; % Time vector of 1 second
f = 10; % Frequency of the sinusoid (Hz)
sinusoid = sin(2*pi*f*t_noisy);

% Create noise in the signal
noise = randn(size(signal));
noisy_signal = [ones_sig, ones_sig, ones_sig, ones_sig, signal + noise*0.1, noise, noise, noise, noise, ones_sig, ones_sig, ones_sig, ones_sig, noise, noise, noise, zeros_sig, zeros_sig, ones_sig, ones_sig, zeros_sig, sinusoid, ones_sig, signal];

% Ensure noisy signal length matches the time vector for plotting
noisy_signal = noisy_signal(1:length(t_noisy));

% Plot the noisy signal
figure;
plot(t_noisy, noisy_signal);
title('Segnale Noisy');
xlabel('Tempo (s)');
ylabel('Ampiezza');
grid on;

% Apply Continuous Wavelet Transform (CWT)
waveletFunction = 'mexh'; % Mexican Hat wavelet
scales = 1:64; % Scales from 1 to 64

% CWT on reference signal
coeff_signal = cwt(signal, scales, waveletFunction);

% CWT on noisy signal
coeff_noisy_signal = cwt(noisy_signal, scales, waveletFunction);

% Cross-correlation of wavelet coefficients
correlation = xcorr(abs(coeff_noisy_signal(:)), abs(coeff_signal(:)));

% Smooth the correlation
smoothed_correlation = smoothdata(abs(correlation), 'gaussian', 50);

% Dynamic threshold based on the mean and standard deviation of the smoothed correlation
mean_corr = mean(smoothed_correlation);
std_corr = std(smoothed_correlation);
threshold = mean_corr + 100 * std_corr;

% Find the indices where the smoothed correlation exceeds the threshold
significant_indices = find(smoothed_correlation > threshold);

% Find the peak of the smoothed correlation
[~, max_index] = max(smoothed_correlation);
lag = max_index - length(noisy_signal);

% Plot CWT of reference signal
figure;
imagesc(t_signal, scales, abs(coeff_signal));
axis xy;
title('Trasformata Wavelet Continua del Segnale di Riferimento');
xlabel('Tempo (s)');
ylabel('Scala');
colorbar;

% Plot CWT of noisy signal
figure;
imagesc(t_noisy, scales, abs(coeff_noisy_signal));
axis xy;
title('Trasformata Wavelet Continua del Segnale Noisy');
xlabel('Tempo (s)');
ylabel('Scala');
colorbar;

% Plot the smoothed correlation and the threshold
figure;
plot(smoothed_correlation);
hold on;
yline(threshold, 'r--', 'Threshold');
title('Smoothed Cross-Correlation with Threshold');
xlabel('Lag');
ylabel('Correlation');
grid on;

% Estimate the start index of the reference signal in the noisy signal
estimated_start_index = lag + length(signal);
if estimated_start_index < 1
    estimated_start_index = 1;
end
if estimated_start_index + length(signal) - 1 > length(noisy_signal)
    estimated_start_index = length(noisy_signal) - length(signal) + 1;
end

% Highlight the found signal in the noisy signal
figure;
plot(t_noisy, noisy_signal);
hold on;
plot(t_noisy(estimated_start_index:estimated_start_index + length(signal) - 1), signal, 'r', 'LineWidth', 2);
title('Segnale Noisy con Segnale di Riferimento Evidenziato');
xlabel('Tempo (s)');
ylabel('Ampiezza');
grid on;

function signal = create_signal(fs, t, amp1, freq1, duration1, start_time1, amp2, freq2, duration2, start_time2, finish_time)
    % Create reference signal
    signal = zeros(size(t));
    
    % First peak
    start_idx1 = ceil(start_time1 * fs) + 1;
    end_idx1 = start_idx1 + round(duration1 * fs) - 1;
    sig1 = amp1 * sin(2 * pi * freq1 * (0:1/fs:finish_time));
    signal(start_idx1:end_idx1) = signal(start_idx1:end_idx1) + sig1(1:length(start_idx1:end_idx1));
    
    % Second peak
    start_idx2 = ceil(start_time2 * fs) + 1;
    end_idx2 = start_idx2 + round(duration2 * fs) - 1;
    sig2 = amp2 * sin(2 * pi * freq2 * (0:1/fs:finish_time));
    signal(start_idx2:end_idx2) = signal(start_idx2:end_idx2) + sig2(1:length(start_idx2:end_idx2));
end

