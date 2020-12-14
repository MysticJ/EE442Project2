function [out_bits] = demodulator(in_syms, modulation_order)
    if ~(modulation_order==1 || modulation_order==2)
        error('Wrong modulation order!')
    end
    if modulation_order == 1
        out_bits = double(in_syms >= 0);
    else
        dis = abs(in_syms - repmat((1/sqrt(2))*[1+1i, -1+1i, 1-1i, -1-1i], size(in_syms, 1), 1));
        [~, min_ind] = min(dis, [], 2);
        bin_num = min_ind-1;
        out_bits = zeros(size(in_syms, 1)*modulation_order, 1);
        for i = 1:size(in_syms, 1)
           switch bin_num(i)
               case 0
                   out_bits(2*i-1:2*i) = [0; 0];
               case 1
                   out_bits(2*i-1:2*i) = [0; 1];
               case 2
                   out_bits(2*i-1:2*i) = [1; 0];
               case 3
                   out_bits(2*i-1:2*i) = [1; 1];
               otherwise
                   out_bits(i:i+1) = [0; 0];
           end
        end
    end
end