% % % % %
% Wireless Receivers: algorithms and architectures
% Audio Transmission Framework 
%
%
%   3 operating modes:
%   - 'matlab' : generic MATLAB audio routines (unreliable under Linux)
%   - 'native' : OS native audio system
%       - ALSA audio tools, most Linux distrubtions
%       - builtin WAV tools on Windows 
%   - 'bypass' : no audio transmission, takes txsignal as received signal

% Configuration Values
conf.audiosystem        = 'bypass';     % Values: 'matlab','native','bypass'
conf.bitsps             = 16;           % bits per audio sample\
conf.nframes            = 1;
conf.modulation_order   = 2;            % BPSK:1, QPSK:2
conf.npreamble          = 100;          % length of preamble

conf.f_spacing          = 5;
conf.nsubcarriers       = 256;
if mod(conf.nsubcarriers,2) ~= 0
   disp('WARNING: Number of subcarriers should be even.'); 
end
conf.os_factor          = 40;
conf.f_sampling         = conf.f_spacing * conf.nsubcarriers * conf.os_factor;
conf.f_mixer            = 2000;
conf.nbits              = 5000;    % number of bits ? per frame?
conf.nsyms              = ceil(conf.nbits/conf.modulation_order);
conf.cp_factor          = 0.5;
conf.lenOFDMsym_cp      = ceil(conf.nsubcarriers*conf.os_factor*(1+conf.cp_factor));
conf.nOFDMsyms          = ceil(conf.nsyms/conf.nsubcarriers);

% Initialize result structure with zero
res.biterrors   = zeros(conf.nframes,1);
res.rxnbits     = zeros(conf.nframes,1);


for k=1:conf.nframes
    
    % Generate random data
    txbits = randi([0 1],conf.nbits,1);
    
    % TODO: Implement tx() Transmit Function
    [txsignal conf] = txtest(txbits,conf,k);
    
    % % % % % % % % % % % %
    % Begin
    % Audio Transmission
    %
    
    % ???? normalize values 
    peakvalue       = max(abs(txsignal));
    normtxsignal    = txsignal / peakvalue;
    % normalize the peak to 1
    
    % create vector for transmission
    % add padding before and after the signal
    rawtxsignal = [zeros(conf.f_sampling,1); normtxsignal; zeros(conf.f_sampling,1)]; 
    % add second channel: no signal
    rawtxsignal = [rawtxsignal, zeros(size(rawtxsignal))]; 
    % calculate length of transmitted signal (in second)
    txdur = length(rawtxsignal)/conf.f_sampling; 
    
%     wavwrite(rawtxsignal,conf.f_s,16,'out.wav')   
    audiowrite('out.wav',rawtxsignal,conf.f_sampling)  
    
    % Platform native audio mode 
    if strcmp(conf.audiosystem,'native')
        
        % Windows WAV mode 
        if ispc()
            disp('Windows WAV');
            wavplay(rawtxsignal,conf.f_sampling,'async');
            disp('Recording in Progress');
            rawrxsignal = wavrecord((txdur+1)*conf.f_sampling,conf.f_sampling);
            disp('Recording complete')
            rxsignal = rawrxsignal(1:end,1);

        % ALSA WAV mode 
        elseif isunix()
            disp('Linux ALSA');
            cmd = sprintf('arecord -c 2 -r %d -f s16_le  -d %d in.wav &',conf.f_sampling,ceil(txdur)+1);
            system(cmd); 
            disp('Recording in Progress');
            system('aplay  out.wav')
            pause(2);
            disp('Recording complete')
            rawrxsignal = wavread('in.wav');
            rxsignal    = rawrxsignal(1:end,1);
        end
        
    % MATLAB audio mode
    elseif strcmp(conf.audiosystem,'matlab')
        disp('MATLAB generic');
        playobj = audioplayer(rawtxsignal,conf.f_sampling,conf.bitsps);
        recobj  = audiorecorder(conf.f_sampling,conf.bitsps,1);
        record(recobj);
        disp('Recording in Progress');
        playblocking(playobj)
        pause(0.5);
        stop(recobj);
        disp('Recording complete')
        rawrxsignal  = getaudiodata(recobj,'int16');
        rxsignal     = double(rawrxsignal(1:end))/double(intmax('int16')) ;
        
    elseif strcmp(conf.audiosystem,'bypass')
        rawrxsignal = rawtxsignal(:,1);
        rxsignal    = rawrxsignal;
    end
    
    % Plot received signal for debgging
    figure(4);
    plot(rawtxsignal(:, 1)); hold on;
    plot(rxsignal); hold off;
    axis([1, length(rxsignal), -0.7, 0.7])
    legend('rawtxsignal', 'rxsignal')
    title('Received Signal')
    
    %
    % End
    % Audio Transmission   
    % % % % % % % % % % % %
    
    % TODO: Implement rx() Receive Function
    % [rxbits conf]       = rx(rxsignal,conf);
    [rxbits conf]       = rxtest(rxsignal, normtxsignal,conf);
    
    res.rxnbits(k)      = length(rxbits);  
    res.biterrors(k)    = sum(rxbits ~= txbits);
    
end

per = sum(res.biterrors > 0)/conf.nframes
ber = sum(res.biterrors)/sum(res.rxnbits)
