function [txsignal conf] = txtest(txbits, conf, frame_ind)

modulation_order    = conf.modulation_order;
os_factor           = conf.os_factor;
nsubcarriers        = conf.nsubcarriers;
npreamble           = conf.npreamble;
nsyms               = conf.nsyms;
cp_factor           = conf.cp_factor;
f_sampling          = conf.f_sampling;
f_mixer             = conf.f_mixer;

%% Genetate & pulse-shape preamble
preamble_bits = preamble_generate(npreamble);
% BPSK
preamble_sym = modulator(preamble_bits, 1);
% upsampling preamble
% preamble_sym = upsample(preamble_sym, os_factor);
preamble_sym = reshape(repmat(preamble_sym.', os_factor, 1), [], 1);
% pulse shaping 
preamble_sym = conv(preamble_sym, rrc(os_factor), 'same');
% % Visualize the preamble
% figure(1);
% stem(preamble_sym(1:200), 'Linewidth', 1); 
% title('example of preamble after pulse shaping');

%% Generate training symbol with preamble
train_sym = train_sym_generate(npreamble, nsubcarriers, modulation_order);

%% fill the empty space of the last block
if (mod(nsyms, nsubcarriers)~=0)
   txbits_fill = ...
       [txbits; randi([0, 1], (nsubcarriers-mod(nsyms, nsubcarriers))*2, 1)]; 
end

%% QPSK modulation
txsyms_ori = modulator(txbits_fill, modulation_order);

%% Reshape to size of (nsubcarriers, []) and add training symbol
txsyms_reshape = [train_sym, reshape(txsyms_ori, nsubcarriers, [])];

%% IFFT
txsyms_ifft = osifft_mod(txsyms_reshape, os_factor);

%% Add cyclic prefix
txsyms_with_cp = ...
    [txsyms_ifft((end-ceil(size(txsyms_ifft, 1)*cp_factor)+1):end, : ); txsyms_ifft];
% % Visualize the signal after ifft
% figure(2);
% subplot(211);
% plot(real(txsyms_with_cp( : , 2)), 'Linewidth', 1); hold on;
% line(nsubcarriers*os_factor/2*ones(1, 2), 0.06*[-1, 1], ...
%     'Color', 'm', 'LineStyle', '--', 'LineWidth', 2); 
% line(nsubcarriers*os_factor*ones(1, 2), 0.06*[-1, 1], ...
%     'Color', 'm', 'LineStyle', '--', 'LineWidth', 2);
% hold off;
% xlabel('index'); ylabel('real part'); title('Real part of an OFDM symbol');
% subplot(212);
% plot(imag(txsyms_with_cp( : , 2)), 'r', 'Linewidth', 1); hold on;
% line(nsubcarriers*os_factor/2*ones(1, 2), 0.045*[-1, 1], ...
%     'Color', '#4DBEEE', 'LineStyle', '--', 'LineWidth', 2); 
% line(nsubcarriers*os_factor*ones(1, 2), 0.045*[-1, 1], ...
%     'Color', '#4DBEEE', 'LineStyle', '--', 'LineWidth', 2);
% hold off;
% xlabel('index'); ylabel('imaginary part'); title('Imaginary part of an OFDM symble');

%% Assemble to signal to transmit
% 
txsyms_ready = [preamble_sym; ...
    mean(abs(preamble_sym), 'all')/mean(abs(txsyms_with_cp), 'all')/2*reshape(txsyms_with_cp, [], 1)];

% txsyms_ready = [preamble_sym; ...
%     max(abs(preamble_sym), [], 'all')/max(abs(txsyms_with_cp), [], 'all')*reshape(txsyms_with_cp, [], 1)];


%% Carrier_sequence
carrier_seq = exp(1i*2*pi*(f_mixer/f_sampling)*(0:length(txsyms_ready)-1));
txsignal = real(txsyms_ready.*(carrier_seq.'));
% % Visualize the transmitted signal
% figure(3);
% subplot(211);
% plot(txsignal(1:nsubcarriers), 'Linewidth', 1); 
% xlabel('1:N'); title('First N samples of txsignal');
% subplot(212);
% plot(txsignal(end-nsubcarriers+1:end), 'r', 'Linewidth', 1); 
% xlabel('Last N'); title('Last N samples of txsignal');

end

