% function stable_combinations = optimize_filter_params(rawData_x, rawData_y, rawData_z, rawData_total, fs, low_freq_vec, high_freq_vec, max_ripple, percH_values, percL_values, attenuation)
%     % Initialize the stable combinations cell array
%     stable_combinations = {};
% 
%     % Loop through all combinations of low and high cutoff frequencies
%     for i = 1:length(low_freq_vec)
%         for j = 1:length(high_freq_vec)
%             low_cutoff_freq = low_freq_vec(i);
%             high_cutoff_freq = high_freq_vec(j);
% 
%             % Initialize variables to track the best combination
%             best_percH = inf;
%             best_percL = -inf;
%             is_stable = false;
% 
%             % Test different percH and percL values
%             for percH = percH_values
%                 for percL = percL_values
%                     try
%                         % Try designing the high-pass and low-pass filters with the current parameters
%                         fprintf('Trying filters with low_cutoff_freq: %f, high_cutoff_freq: %f, percH: %f, percL: %f\n', low_cutoff_freq, high_cutoff_freq, percH, percL);
%                         [b_high, a_high] = cheby2_highpass_filter(fs, high_cutoff_freq, max_ripple, percH, attenuation);
%                         [b_low, a_low] = cheby2_lowpass_filter(fs, low_cutoff_freq, max_ripple, percL, attenuation);
% 
%                         % Check for filter stability using MATLAB's isstable function
%                         if isstable(b_high, a_high) && isstable(b_low, a_low)
%                             % Apply the high-pass filter first
%                             filtData_x = filtfilt(b_high, a_high, rawData_x);
%                             filtData_y = filtfilt(b_high, a_high, rawData_y);
%                             filtData_z = filtfilt(b_high, a_high, rawData_z);
%                             filtData_total = filtfilt(b_high, a_high, rawData_total);
% 
%                             % Then apply the low-pass filter
%                             filtData_x = filtfilt(b_low, a_low, filtData_x);
%                             filtData_y = filtfilt(b_low, a_low, filtData_y);
%                             filtData_z = filtfilt(b_low, a_low, filtData_z);
%                             filtData_total = filtfilt(b_low, a_low, filtData_total);
% 
%                             % If successful, update the best percH and percL
%                             if percH < best_percH || (percH == best_percH && percL > best_percL)
%                                 best_percH = percH;
%                                 best_percL = percL;
%                                 is_stable = true;
%                             end
%                         end
%                     catch ME
%                         % If the filter fails, continue to the next combination
%                         fprintf('Filter failed for low_cutoff_freq: %f, high_cutoff_freq: %f, percH: %f, percL: %f. Error: %s\n', low_cutoff_freq, high_cutoff_freq, percH, percL, ME.message);
%                         continue;
%                     end
%                 end
%             end
% 
%             % If a stable combination was found, store it
%             if is_stable
%                 stable_combinations{end+1} = struct('low_cutoff_freq', low_cutoff_freq, 'high_cutoff_freq', high_cutoff_freq, 'max_ripple', max_ripple, 'percH', best_percH, 'percL', best_percL);
%             end
%         end
%     end
% end
% 
% function [b_high, a_high] = cheby2_highpass_filter(fs, high_cutoff_freq, max_ripple, percH, attenuation)
%     % Calculate the passband and stopband edges without margin
%     passband = high_cutoff_freq / (fs / 2);
%     stopband = (high_cutoff_freq * percH) / (fs / 2);
% 
%     fprintf('High-Pass Filter - Passband: [%f], Stopband: [%f]\n', passband, stopband);
% 
%     % Design a Chebyshev Type II high-pass filter with the given parameters
%     [n, Ws] = cheb2ord(passband, stopband, max_ripple, attenuation);
%     [b_high, a_high] = cheby2(ceil(n), max_ripple, Ws, 'high');  % Ensure n is an integer
% end
% 
% function [b_low, a_low] = cheby2_lowpass_filter(fs, low_cutoff_freq, max_ripple, percL, attenuation)
%     % Calculate the passband and stopband edges without margin
%     passband = low_cutoff_freq / (fs / 2);
%     stopband = (low_cutoff_freq * percL) / (fs / 2);
% 
%     fprintf('Low-Pass Filter - Passband: [%f], Stopband: [%f]\n', passband, stopband);
% 
%     % Design a Chebyshev Type II low-pass filter with the given parameters
%     [n, Ws] = cheb2ord(passband, stopband, max_ripple, attenuation);
%     [b_low, a_low] = cheby2(ceil(n), max_ripple, Ws, 'low');  % Ensure n is an integer
% end

% function ranked_combinations = optimize_filter_params(rawData_x, rawData_y, rawData_z, rawData_total, fs, low_freq_vec, high_freq_vec, max_ripple, percH_values, percL_values, attenuation, prototype_filter)
%     % Initialize the stable combinations cell array
%     stable_combinations = {};
% 
%     % Loop through all combinations of low and high cutoff frequencies
%     for i = 1:length(low_freq_vec)
%         for j = 1:length(high_freq_vec)
%             low_cutoff_freq = low_freq_vec(i);
%             high_cutoff_freq = high_freq_vec(j);
% 
%             % Initialize variables to track the best combination
%             best_percH = inf;
%             best_percL = -inf;
%             is_stable = false;
% 
%             % Test different percH and percL values
%             for percH = percH_values
%                 for percL = percL_values
%                     try
%                         % Try designing the high-pass and low-pass filters with the current parameters
%                         fprintf('Trying filters with low_cutoff_freq: %f, high_cutoff_freq: %f, percH: %f, percL: %f\n', low_cutoff_freq, high_cutoff_freq, percH, percL);
%                         [b_high, a_high] = cheby2_highpass_filter(fs, high_cutoff_freq, max_ripple, percH, attenuation);
%                         [b_low, a_low] = cheby2_lowpass_filter(fs, low_cutoff_freq, max_ripple, percL, attenuation);
% 
%                         % Check for filter stability using MATLAB's isstable function
%                         if isstable(b_high, a_high) && isstable(b_low, a_low)
%                             % Apply the high-pass filter first
%                             filtData_x = filtfilt(b_high, a_high, rawData_x);
%                             filtData_y = filtfilt(b_high, a_high, rawData_y);
%                             filtData_z = filtfilt(b_high, a_high, rawData_z);
%                             filtData_total = filtfilt(b_high, a_high, rawData_total);
% 
%                             % Then apply the low-pass filter
%                             filtData_x = filtfilt(b_low, a_low, filtData_x);
%                             filtData_y = filtfilt(b_low, a_low, filtData_y);
%                             filtData_z = filtfilt(b_low, a_low, filtData_z);
%                             filtData_total = filtfilt(b_low, a_low, filtData_total);
% 
%                             % If successful, update the best percH and percL
%                             if percH < best_percH || (percH == best_percH && percL > best_percL)
%                                 best_percH = percH;
%                                 best_percL = percL;
%                                 is_stable = true;
%                             end
%                         end
%                     catch ME
%                         % If the filter fails, continue to the next combination
%                         fprintf('Filter failed for low_cutoff_freq: %f, high_cutoff_freq: %f, percH: %f, percL: %f. Error: %s\n', low_cutoff_freq, high_cutoff_freq, percH, percL, ME.message);
%                         continue;
%                     end
%                 end
%             end
% 
%             % If a stable combination was found, store it
%             if is_stable
%                 stable_combinations{end+1} = struct('low_cutoff_freq', low_cutoff_freq, 'high_cutoff_freq', high_cutoff_freq, 'max_ripple', max_ripple, 'percH', best_percH, 'percL', best_percL);
%             end
%         end
%     end
% 
%     % Rank the combinations based on the sum of the distances to the prototype filter stopbands
%     prototype_low_stopband = prototype_filter.low_stopband;
%     prototype_high_stopband = prototype_filter.high_stopband;
%     ranking_scores = [];
% 
%     for k = 1:length(stable_combinations)
%         combination = stable_combinations{k};
%         low_stopband = combination.low_cutoff_freq * combination.percL;
%         high_stopband = combination.high_cutoff_freq * combination.percH;
% 
%         low_stopband_diff = abs(low_stopband - prototype_low_stopband);
%         high_stopband_diff = abs(high_stopband - prototype_high_stopband);
% 
%         total_diff = low_stopband_diff + high_stopband_diff;
%         ranking_scores = [ranking_scores; total_diff, k];
%     end
% 
%     % Sort the combinations by their ranking scores
%     ranking_scores = sortrows(ranking_scores);
% 
%     % Create the ranked combinations matrix
%     ranked_combinations = [];
%     for idx = 1:size(ranking_scores, 1)
%         k = ranking_scores(idx, 2);
%         combination = stable_combinations{k};
%         ranked_combinations = [ranked_combinations; combination.low_cutoff_freq, combination.high_cutoff_freq, combination.max_ripple, combination.percH, combination.percL];
%     end
% end
% 
% function [b_high, a_high] = cheby2_highpass_filter(fs, high_cutoff_freq, max_ripple, percH, attenuation)
%     % Calculate the passband and stopband edges without margin
%     passband = high_cutoff_freq / (fs / 2);
%     stopband = (high_cutoff_freq * percH) / (fs / 2);
% 
%     fprintf('High-Pass Filter - Passband: [%f], Stopband: [%f]\n', passband, stopband);
% 
%     % Design a Chebyshev Type II high-pass filter with the given parameters
%     [n, Ws] = cheb2ord(passband, stopband, max_ripple, attenuation);
%     [b_high, a_high] = cheby2(ceil(n), max_ripple, Ws, 'high');  % Ensure n is an integer
% end
% 
% function [b_low, a_low] = cheby2_lowpass_filter(fs, low_cutoff_freq, max_ripple, percL, attenuation)
%     % Calculate the passband and stopband edges without margin
%     passband = low_cutoff_freq / (fs / 2);
%     stopband = (low_cutoff_freq * percL) / (fs / 2);
% 
%     fprintf('Low-Pass Filter - Passband: [%f], Stopband: [%f]\n', passband, stopband);
% 
%     % Design a Chebyshev Type II low-pass filter with the given parameters
%     [n, Ws] = cheb2ord(passband, stopband, max_ripple, attenuation);
%     [b_low, a_low] = cheby2(ceil(n), max_ripple, Ws, 'low');  % Ensure n is an integer
% end

function [ranked_combinations, filtData_x, filtData_y, filtData_z, filtData_total] = optimize_filter_params(rawData_x, rawData_y, rawData_z, rawData_total, fs, low_freq_vec, high_freq_vec, max_ripple, percH_values, percL_values, attenuation, prototype_filter, verbose)
    % Initialize the stable combinations cell array
    stable_combinations = {};

    % Loop through all combinations of low and high cutoff frequencies
    for i = 1:length(low_freq_vec)
        for j = 1:length(high_freq_vec)
            low_cutoff_freq = low_freq_vec(i);
            high_cutoff_freq = high_freq_vec(j);

            % Initialize variables to track the best combination
            best_percH = inf;
            best_percL = -inf;
            is_stable = false;

            % Test different percH and percL values
            for percH = percH_values
                for percL = percL_values
                    try
                        % Try designing the high-pass and low-pass filters with the current parameters
                        fprintf('Trying filters with low_cutoff_freq: %f, high_cutoff_freq: %f, percH: %f, percL: %f\n', low_cutoff_freq, high_cutoff_freq, percH, percL);
                        [b_high, a_high] = cheby2_highpass_filter(fs, high_cutoff_freq, max_ripple, percH, attenuation);
                        [b_low, a_low] = cheby2_lowpass_filter(fs, low_cutoff_freq, max_ripple, percL, attenuation);

                        % Check for filter stability using MATLAB's isstable function
                        if isstable(b_high, a_high) && isstable(b_low, a_low)
                            % Apply the high-pass filter first
                            filtData_x = filtfilt(b_high, a_high, rawData_x);
                            filtData_y = filtfilt(b_high, a_high, rawData_y);
                            filtData_z = filtfilt(b_high, a_high, rawData_z);
                            filtData_total = filtfilt(b_high, a_high, rawData_total);
                            
                            % Then apply the low-pass filter
                            filtData_x = filtfilt(b_low, a_low, filtData_x);
                            filtData_y = filtfilt(b_low, a_low, filtData_y);
                            filtData_z = filtfilt(b_low, a_low, filtData_z);
                            filtData_total = filtfilt(b_low, a_low, filtData_total);

                            % If successful, update the best percH and percL
                            if percH < best_percH || (percH == best_percH && percL > best_percL)
                                best_percH = percH;
                                best_percL = percL;
                                is_stable = true;
                            end
                        end
                    catch ME
                        % If the filter fails, continue to the next combination
                        fprintf('Filter failed for low_cutoff_freq: %f, high_cutoff_freq: %f, percH: %f, percL: %f. Error: %s\n', low_cutoff_freq, high_cutoff_freq, percH, percL, ME.message);
                        continue;
                    end
                end
            end

            % If a stable combination was found, store it
            if is_stable
                stable_combinations{end+1} = struct('low_cutoff_freq', low_cutoff_freq, 'high_cutoff_freq', high_cutoff_freq, 'max_ripple', max_ripple, 'percH', best_percH, 'percL', best_percL);
            end
        end
    end

    % Rank the combinations based on the sum of the distances to the prototype filter stopbands
    prototype_low_stopband = prototype_filter.low_stopband;
    prototype_high_stopband = prototype_filter.high_stopband;
    ranking_scores = [];

    for k = 1:length(stable_combinations)
        combination = stable_combinations{k};
        low_stopband = combination.low_cutoff_freq * combination.percL;
        high_stopband = combination.high_cutoff_freq * combination.percH;

        low_stopband_diff = abs(low_stopband - prototype_low_stopband);
        high_stopband_diff = abs(high_stopband - prototype_high_stopband);

        total_diff = low_stopband_diff + high_stopband_diff;
        ranking_scores = [ranking_scores; total_diff, k];
    end

    % Sort the combinations by their ranking scores
    ranking_scores = sortrows(ranking_scores);

    % Create the ranked combinations matrix
    ranked_combinations = [];
    best_combination = [];
    for idx = 1:size(ranking_scores, 1)
        k = ranking_scores(idx, 2);
        combination = stable_combinations{k};
        ranked_combinations = [ranked_combinations; combination.low_cutoff_freq, combination.high_cutoff_freq, combination.max_ripple, combination.percH, combination.percL];
        if idx == 1
            best_combination = combination;
        end
    end

    % Apply the best combination to filter the data
    if ~isempty(best_combination)
        [b_high, a_high] = cheby2_highpass_filter(fs, best_combination.high_cutoff_freq, best_combination.max_ripple, best_combination.percH, attenuation);
        [b_low, a_low] = cheby2_lowpass_filter(fs, best_combination.low_cutoff_freq, best_combination.max_ripple, best_combination.percL, attenuation);

        % Apply the high-pass filter first
        filtData_x = filtfilt(b_high, a_high, rawData_x);
        filtData_y = filtfilt(b_high, a_high, rawData_y);
        filtData_z = filtfilt(b_high, a_high, rawData_z);
        filtData_total = filtfilt(b_high, a_high, rawData_total);
        
        % Then apply the low-pass filter
        filtData_x = filtfilt(b_low, a_low, filtData_x);
        filtData_y = filtfilt(b_low, a_low, filtData_y);
        filtData_z = filtfilt(b_low, a_low, filtData_z);
        filtData_total = filtfilt(b_low, a_low, filtData_total);

        % Print the parameters used for filtering
        fprintf('Best combination used for filtering: low_cutoff_freq: %f, high_cutoff_freq: %f, percH: %f, percL: %f\n', best_combination.low_cutoff_freq, best_combination.high_cutoff_freq, best_combination.percH, best_combination.percL);
    else
        error('No stable combination found');
    end
end

function [b_high, a_high] = cheby2_highpass_filter(fs, high_cutoff_freq, max_ripple, percH, attenuation)
    % Calculate the passband and stopband edges without margin
    passband = high_cutoff_freq / (fs / 2);
    stopband = (high_cutoff_freq * percH) / (fs / 2);
    
    fprintf('High-Pass Filter - Passband: [%f], Stopband: [%f]\n', passband, stopband);

    % Design a Chebyshev Type II high-pass filter with the given parameters
    [n, Ws] = cheb2ord(passband, stopband, max_ripple, attenuation);
    [b_high, a_high] = cheby2(ceil(n), max_ripple, Ws, 'high');  % Ensure n is an integer
end

function [b_low, a_low] = cheby2_lowpass_filter(fs, low_cutoff_freq, max_ripple, percL, attenuation)
    % Calculate the passband and stopband edges without margin
    passband = low_cutoff_freq / (fs / 2);
    stopband = (low_cutoff_freq * percL) / (fs / 2);
    
    fprintf('Low-Pass Filter - Passband: [%f], Stopband: [%f]\n', passband, stopband);

    % Design a Chebyshev Type II low-pass filter with the given parameters
    [n, Ws] = cheb2ord(passband, stopband, max_ripple, attenuation);
    [b_low, a_low] = cheby2(ceil(n), max_ripple, Ws, 'low');  % Ensure n is an integer
end

