function [train_sym] = train_sym_generate(npreamble, nsubcarriers, modulation_order)
% train_sym_ori = modulator(...
%     [preamble_generate(npreamble); ...
%     zeros(nsubcarriers*modulation_order-npreamble, 1)], ...
%     modulation_order);
preamble_bits = preamble_generate(npreamble);
if (npreamble >= nsubcarriers*modulation_order)
    train_bits = preamble_bits(1:nsubcarriers*modulation_order);
else
    temp = repmat(preamble_bits, ceil(nsubcarriers*modulation_order/npreamble), 1);
    train_bits = temp(1:nsubcarriers*modulation_order);
end
train_sym = modulator(train_bits, modulation_order);
end

