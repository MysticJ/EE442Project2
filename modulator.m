function out_syms = modulator(in_bits, modulation_order)

if (modulation_order~=1 && modulation_order~=2)
    error('Modulation_order wrong!')
end

if modulation_order == 1 % BPSK
    out_syms = 2*in_bits-1; % 0 -> -1; 1 -> 0 
else % QPSK
    out_syms = zeros(ceil(size(in_bits, 1)/modulation_order), 1);
    for i = 1:size(out_syms, 1)
       temp = 2*in_bits(2*i-1)+in_bits(2*i);
       switch temp
           case 0
               out_syms(i) = 1/sqrt(2)*(1+1i);
           case 1
               out_syms(i) = 1/sqrt(2)*(-1+1i);
           case 2
               out_syms(i) = 1/sqrt(2)*(1-1i);
           case 3
               out_syms(i) = 1/sqrt(2)*(-1-1i);
           otherwise
               out_syms(i) = 0;
       end
    end
end

end