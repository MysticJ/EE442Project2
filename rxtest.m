function [rxbits, rxbits_pt, conf]       = rxtest(rxsignal, rawtxsignal, conf)
f_mixer         = conf.f_mixer;
f_sampling      = conf.f_sampling;
os_factor       = conf.os_factor;
nsubcarriers    = conf.nsubcarriers;
% f_spacing       = conf.f_spacing;
cp_factor       = conf.cp_factor;
nsyms           = conf.nsyms;
lenOFDMsym_cp   = conf.lenOFDMsym_cp;
nOFDMsyms       = conf.nOFDMsyms;
npreamble       = conf.npreamble;
modulation_order = conf.modulation_order;

%% Down conversion
carrier_seq = exp(-1i*2*pi*(f_mixer/f_sampling)*(0:length(rxsignal)-1)).';
rxsignal_dc = rxsignal.*carrier_seq;

% % txsignal
tx_carrier_seq = exp(-1i*2*pi*(f_mixer/f_sampling)*(0:length(rawtxsignal)-1)).';
txsignal_dc = rawtxsignal.*tx_carrier_seq;

%% Lowpass filter
rxsignal_lp = ofdmlowpass_mod(rxsignal_dc, f_sampling, nsubcarriers*os_factor);

% % txsignal
txsignal_lp = ofdmlowpass_mod(txsignal_dc, f_sampling, nsubcarriers*os_factor);

% % Visualize effect of low-pass filter in bypass mode
% % !!! Index only valid in bypass mode
% figure(5); 
% subplot(211);
% plot(real(txsignal_dc(f_sampling:f_sampling+150)), 'LineWidth', 2);
% title('e.g. real part of rxsignal before lowpass filter')
% subplot(212); 
% plot(real(txsignal_lp(f_sampling:f_sampling+150)), 'Color', '#D95319', 'LineWidth', 2);
% title('e.g. real part of rxsignal after lowpass filter');

%% Frame synchronization in time domain
data_idx = frame_sync(rxsignal_lp, npreamble, os_factor);
% temp = data_idx+(nOFDMsyms+1)*lenOFDMsym_cp-1
% data_length = length(rxsignal_lp)
% assert(data_idx+(nOFDMsyms+1)*lenOFDMsym_cp-1 < length(rxsignal_lp));
if (data_idx+(nOFDMsyms+1)*lenOFDMsym_cp-1 >= length(rxsignal_lp))
   rxbits = randi([0, 1], nsyms*modulation_order, 1);
   rxbits_pt = rxbits;
   disp('Incorrect starting index detected.');
   return
end
OFDM_sym_with_train_cp = reshape(...
    rxsignal_lp(...
    data_idx : data_idx+(nOFDMsyms+1)*lenOFDMsym_cp-1 ...
    ), ...
    lenOFDMsym_cp, []);

% % txsignal
tx_data_idx = frame_sync(txsignal_lp, npreamble, os_factor);
% assert(tx_data_idx+(nOFDMsyms+1)*lenOFDMsym_cp-1 < length(txsignal_lp), 'Incorrect starting index');
if (tx_data_idx+(nOFDMsyms+1)*lenOFDMsym_cp-1 >= length(txsignal_lp))
   rxbits = randi([0, 1], nsyms*modulation_order, 1);
   rxbits_pt = rxbits;
   disp('Incorrect starting index detected for txsignal');
   return
end
tx_OFDM_sym_with_train_cp = reshape(...
    txsignal_lp(...
    tx_data_idx : tx_data_idx+(nOFDMsyms+1)*lenOFDMsym_cp-1 ...
    ), ...
    lenOFDMsym_cp, []);

% % Visualize rxsignal and txsignal before FFT
% figure(6);
% subplot(211);
% plot(abs(tx_OFDM_sym_with_train_cp(1:500, 2)), 'LineWidth', 2); hold on;
% plot(abs(OFDM_sym_with_train_cp(1:500, 2)), 'LineWidth', 2); hold off;
% ylabel('Magnitude'); legend('txsignal', 'rxsignal'); 
% title('One OFDM symbol of txsignal and rxsignal');
% subplot(212);
% plot(angle(tx_OFDM_sym_with_train_cp(1:500, 2)), 'LineWidth', 2); hold on;
% plot(angle(OFDM_sym_with_train_cp(1:500, 2)), 'LineWidth', 2); hold off;
% ylabel('Phase'); legend('txsignal', 'rxsignal'); 
% title('One OFDM symbol of txsignal and rxsignal');

%% Plot spectrum
% % Visualize spectrum by symbol after down-conversion
conf.delay_factor = ...
    plot_spectrum_sym(OFDM_sym_with_train_cp, tx_OFDM_sym_with_train_cp, conf, false)

%% Remove cyclic prefix
OFDM_sym_with_train = ...
    OFDM_sym_with_train_cp(ceil(nsubcarriers*os_factor*cp_factor)+1:end, : );

% % txsignal
tx_OFDM_sym_with_train = ...
    tx_OFDM_sym_with_train_cp(ceil(nsubcarriers*os_factor*cp_factor)+1:end, : );

%% FFT
OFDM_sym_train_fft = osfft_mod(OFDM_sym_with_train, os_factor);

% % txsignal
tx_OFDM_sym_train_fft = osfft_mod(tx_OFDM_sym_with_train, os_factor);

% % Visualize rxsignal and txsignal after FFT
% figure(8);
% subplot(211);
% plot(abs(tx_OFDM_sym_train_fft(:, 2)), 'LineWidth', 2); hold on;
% plot(abs(OFDM_sym_train_fft(:, 2)), 'LineWidth', 2); hold off;
% ylabel('Magnitude'); legend('txsignal', 'rxsignal'); 
% title('FFT of One OFDM symbol  - Magnitude');
% subplot(212);
% plot(angle(tx_OFDM_sym_train_fft(:, 2)), 'LineWidth', 2); hold on;
% plot(angle(OFDM_sym_train_fft(:, 2)), 'LineWidth', 2); hold off;
% ylabel('Phase'); legend('txsignal', 'rxsignal'); 
% title('FFT of One OFDM symbol  - Phase');

%% Separate training and data symbol
train_sym_fft = OFDM_sym_train_fft( : , 1);
OFDM_sym_fft = OFDM_sym_train_fft( : , 2:end);

% % txsignal
tx_train_sym_fft = tx_OFDM_sym_train_fft( : , 1);
tx_OFDM_sym_fft = tx_OFDM_sym_train_fft( : , 2:end);

%% Channel equalizer
% % generate training symbol with preamble(same as in transmitter)
train_sym_ori = train_sym_generate(npreamble, nsubcarriers, 1);
% % get channel response (estimated)
channel_response = train_sym_fft./train_sym_ori;
% % remove channel response
OFDM_sym_equ = OFDM_sym_fft./repmat(channel_response, 1, size(OFDM_sym_fft, 2));

% % txsignal
tx_channel_response = tx_train_sym_fft./train_sym_ori;
tx_OFDM_sym_equ = tx_OFDM_sym_fft./repmat(tx_channel_response, 1, size(tx_OFDM_sym_fft, 2));

% % Visualize channel response
% figure(9);
% subplot(211);
% plot(abs(tx_channel_response), 'LineWidth', 2); hold on;
% plot(abs(channel_response), 'LineWidth', 2); hold off;
% ylabel('Magnitude'); legend('txsignal', 'rxsignal'); 
% title('Channel response  - Magnitude');
% subplot(212);
% plot(angle(tx_channel_response), 'LineWidth', 2); hold on;
% plot(angle(channel_response), 'LineWidth', 2); hold off;
% ylabel('Phase'); legend('txsignal', 'rxsignal'); 
% title('Channel response  - Phase');

% % Visualize symbol magnitude and phase after equalizer
% figure(10);
% subplot(211);
% plot(abs(tx_OFDM_sym_equ(:, end)), 'LineWidth', 2); hold on;
% plot(abs(OFDM_sym_equ(:, end)), 'LineWidth', 2); hold off;
% ylabel('Magnitude'); legend('txsignal', 'rxsignal'); 
% title('One OFDM symbol after equalizer - Magnitude');
% subplot(212);
% plot(angle(tx_OFDM_sym_equ(:, end)), 'LineWidth', 2); hold on;
% plot(angle(OFDM_sym_equ(:, end)), 'LineWidth', 2); hold off;
% ylabel('Phase'); legend('txsignal', 'rxsignal'); 
% title('One OFDM symbol after equalizer - Phase');

%% % Visualize channel response of each symbol
% for ii = 1:nOFDMsyms
%    temp = OFDM_sym_equ(:, ii)./tx_OFDM_sym_equ(:, ii);
%    figure(11); 
%    subplot(211); 
%    plot(abs(temp), 'LineWidth', 1.5, 'Color', '#0072BD'); 
%    axis([1, nsubcarriers, 0, 2]); ylabel('magnitude'); legend('Magnitude');
%    subplot(212); 
%    plot(angle(temp), 'LineWidth', 1.5, 'Color', '#77AC30'); 
%    axis([1, nsubcarriers, -pi, pi]); ylabel('phase');legend('Phase');
%    pause(0.5);
% end

%% Time domain channel response
% channel_response_ifft = osifft_mod(channel_response, os_factor);
% channel_response_ifft = ifft(channel_response, os_factor);
% figure(12);
% subplot(211);
% plot(abs(channel_response_ifft));
% subplot(212);
% plot(angle(channel_response_ifft));

%% Phase Tracking
theta_hat = zeros(nsubcarriers, nOFDMsyms+1);
theta_hat(:, 1) = angle(-train_sym_ori.^4)/4;
OFDM_sym_pt = zeros(size(OFDM_sym_equ)); 

for ii = 1:nOFDMsyms
    deltaTheta = repmat(angle(-OFDM_sym_equ(:, ii).^4)/4, 1, 6) + ...
        repmat(pi/2*(-1:4), size(OFDM_sym_equ, 1), 1);
    [~, ind] = min(abs(deltaTheta-repmat(theta_hat(:, 1), 1, 6)), [], 2);
    theta = deltaTheta(sub2ind(size(deltaTheta), (1:nsubcarriers).', ind));
    theta_hat(:, ii+1) = wrapToPi(0.5*theta + 0.5*theta_hat(:, ii));
    OFDM_sym_pt(:, ii) = OFDM_sym_equ(:, ii).*exp(-1i*theta_hat(:, ii+1));
end

% % Visualize phase shift of some channels after phase tracking
% figure(13);
% subplot(311);
% plot(wrapToPi(angle(-tx_OFDM_sym_equ(1, :).^4)/4)/pi/2, ...
%     'LineWidth', 2, 'LineStyle', '-.', 'Color', '#0072BD'); hold on;
% plot(wrapToPi(angle(-OFDM_sym_equ(1, :).^4)/4)/pi/2, ...
%     'LineWidth', 2, 'Color', '#77AC30');
% plot(wrapToPi(angle(-OFDM_sym_pt(1, :).^4)/4)/pi/2, ...
%     'LineWidth', 2, 'Color', '#D95319'); hold off;
% ylabel('Phase shift/ 2\pi');
% title('Phase shift at first subcarrier');
% legend('txsignal', 'before phase correction', 'after phase correction', 'Location', 'northwest');
% subplot(312);
% plot(wrapToPi(angle(-tx_OFDM_sym_equ(round(size(OFDM_sym_equ, 1)/2), :).^4)/4)/pi/2, ...
%     'LineWidth', 2, 'LineStyle', '-.', 'Color', '#0072BD'); hold on;
% plot(wrapToPi(angle(-OFDM_sym_equ(round(size(OFDM_sym_equ, 1)/2), :).^4)/4)/pi/2, ...
%     'LineWidth', 2, 'Color', '#77AC30');
% plot(wrapToPi(angle(-OFDM_sym_pt(round(size(OFDM_sym_equ, 1)/2), :).^4)/4)/pi/2, ...
%     'LineWidth', 2, 'Color', '#D95319'); hold off;
% ylabel('Phase shift/ 2\pi');
% title('Phase shift at mid subcarrier');
% legend('txsignal', 'before phase correction', 'after phase correction', 'Location', 'northwest');
% subplot(313);
% plot(wrapToPi(angle(-tx_OFDM_sym_equ(end, :).^4)/4)/pi/2, ...
%     'LineWidth', 2, 'LineStyle', '-.', 'Color', '#0072BD'); hold on;
% plot(wrapToPi(angle(-OFDM_sym_equ(end, :).^4)/4)/pi/2, ...
%     'LineWidth', 2, 'Color', '#77AC30');
% plot(wrapToPi(angle(-OFDM_sym_pt(end, :).^4)/4)/pi/2, ...
%     'LineWidth', 2, 'Color', '#D95319'); hold off;
% ylabel('Phase shift/ 2\pi');
% title('Phase shift at last subcarrier');
% legend('txsignal', 'before phase correction', 'after phase correction', 'Location', 'northwest');


%% Demodulation
OFDM_sym_col = reshape(OFDM_sym_equ, [], 1);
rxbits = demodulator(OFDM_sym_col(1:nsyms), modulation_order);

OFDM_sym_col = reshape(OFDM_sym_pt, [], 1);
rxbits_pt = demodulator(OFDM_sym_col(1:nsyms), modulation_order);

end

function [beginning_of_data] = frame_sync(in_syms, npreamble, os_factor)

preamble_syms = modulator(preamble_generate(npreamble), 1);
current_peak_value = 0;
samples_after_threshold = os_factor;
detection_threshold = 15;
beginning_of_data = 0;
% corVal = zeros(length(in_syms)-os_factor*npreamble, 1);

for i = os_factor*npreamble+1:length(in_syms)
    r = in_syms(i-os_factor*npreamble:os_factor:i-os_factor); 
    c = preamble_syms'*r;
    T = abs(c)^2/abs((r')*r);
    
    % corVal(i-os_factor*npreamble) = T; 
    
    if (T > detection_threshold || samples_after_threshold < os_factor)
        samples_after_threshold = samples_after_threshold - 1;
        if (T > current_peak_value)
            beginning_of_data = i;
            % phase_of_peak = mod(angle(c),2*pi);
            % magnitude_of_peak = abs(c)/npreamble;
            current_peak_value = T;
        end
    end
end
% figure(3);plot(corVal); ylabel('covariance with preamble')
assert(beginning_of_data ~= 0, 'No correct preamble detected');
end

function [delay_factor] = plot_spectrum_sym(data, ref_data, conf, to_display)
    temp = fft(data, [], 1);
    ref_temp = fft(ref_data, [], 1);
    response_temp = ifft(temp./ref_temp, [], 1);
    [~, ind] = max(response_temp(1:size(response_temp, 1)/2, :), [], 1);
    delay_factor = max(ind)/conf.nsubcarriers/conf.os_factor;
    if (to_display)
        f_sampling = conf.f_sampling;
        data_len = size(data, 1);
        idx = (-data_len/2:data_len/2-1)*(f_sampling/data_len);
        max_temp = max(abs(temp), [], 'all');
        figure(7);
        for ii = 1:size(data, 2)
            subplot(2,3, [1, 2]);
            plot(idx, abs(fftshift(ref_temp(:, ii))), 'LineWidth', 1.5); hold on;
            plot(idx, abs(fftshift(temp(:, ii))), 'LineWidth', 1); hold off;
            axis([-f_sampling/6, f_sampling/6, 0, max_temp]);
            title('Spectrum - Magnitude'); legend('txsignal', 'rxsignal');
            subplot(2,3, [4, 5]);
            plot(idx, angle(fftshift(ref_temp(:, ii))), 'LineWidth', 1.5); hold on;
            plot(idx, angle(fftshift(temp(:, ii))), 'LineWidth', 1); hold off;
            axis([-f_sampling/6, f_sampling/6, -3.15, 3.15]);
            title('Spectrum - Phase'); legend('txsignal', 'rxsignal');
            subplot(2, 3, [3, 6]);
            plot(abs(response_temp(:, ii)), 'LineWidth', 1.5);
            axis([-1000, ceil(conf.nsubcarriers*conf.os_factor*conf.cp_factor), 0, max(abs(response_temp(:, ii)))]);
            title('IFFT of channel response'); 
            pause(0.5);
        end
    end
end
