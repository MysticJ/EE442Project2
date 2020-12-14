function [rxbits conf]       = rxtest(rxsignal, normtxsignal,conf)
f_mixer         = conf.f_mixer;
f_sampling      = conf.f_sampling;
os_factor       = conf.os_factor;
nsubcarriers    = conf.nsubcarriers;
f_spacing       = conf.f_spacing;
cp_factor       = conf.cp_factor;
nsyms           = conf.nsyms;
lenOFDMsym_cp   = conf.lenOFDMsym_cp;
nOFDMsyms       = conf.nOFDMsyms;
npreamble       = conf.npreamble;
modulation_order = conf.modulation_order;

%% Down conversion
carrier_seq = exp(-1i*2*pi*(f_mixer/f_sampling)*(0:length(rxsignal)-1)).';
rxsignal_dc = rxsignal.*carrier_seq;

%% Lowpass filter
rxsignal_lp = ofdmlowpass_mod(rxsignal_dc, f_sampling, nsubcarriers*os_factor);
% % Visualize effect of low-pass filter in bypass mode
% % !!! Index only valid in bypass mode
% figure(5); 
% subplot(211);
% plot(real(rxsignal_dc(f_sampling:f_sampling+150)), 'LineWidth', 2);
% title('e.g. real part of rxsignal before lowpass filter')
% subplot(212); 
% plot(real(rxsignal_lp(f_sampling:f_sampling+150)), 'Color', '#D95319', 'LineWidth', 2);
% title('e.g. real part of rxsignal after lowpass filter');

%% Frame synchronization in time domain
data_idx = frame_sync(rxsignal_lp, npreamble, os_factor);
% temp = data_idx+(nOFDMsyms+1)*lenOFDMsym_cp-1
% data_length = length(rxsignal_lp)
assert(data_idx+(nOFDMsyms+1)*lenOFDMsym_cp-1 < length(rxsignal_lp));
OFDM_sym_with_train_cp = reshape(...
    rxsignal_lp(...
    data_idx : data_idx+(nOFDMsyms+1)*lenOFDMsym_cp-1 ...
    ), ...
    lenOFDMsym_cp, []);

%% Remove cyclic prefix
OFDM_sym_with_train = ...
    OFDM_sym_with_train_cp(ceil(nsubcarriers*os_factor*cp_factor)+1:end, : );

%% FFT
OFDM_sym_train_fft = osfft_mod(OFDM_sym_with_train, os_factor);

%% Separate training and data symbol
train_sym_fft = OFDM_sym_train_fft( : , 1);
OFDM_sym_fft = OFDM_sym_train_fft( : , 2:end);

%% Channel equalizer
% % generate training symbol with preamble(same as in transmitter)
train_sym_ori = train_sym_generate(npreamble, nsubcarriers, modulation_order);
% % get channel response (estimated)
channel_response = train_sym_fft./train_sym_ori;
% % remove channel response
OFDM_sym_equ = OFDM_sym_fft./repmat(channel_response, 1, size(OFDM_sym_fft, 2));

%% Demodulation
OFDM_sym_col = reshape(OFDM_sym_equ, [], 1);
rxbits = demodulator(OFDM_sym_col(1:nsyms), modulation_order);

end

function [beginning_of_data] = frame_sync(in_syms, npreamble, os_factor)

preamble_syms = modulator(preamble_generate(npreamble), 1);
current_peak_value = 0;
samples_after_threshold = os_factor;
detection_threshold = 15;
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
end