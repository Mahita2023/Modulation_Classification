function modulator = classGetModulator(modType, sps, ~)
switch modType
  case "BPSK"
    modulator = @(x)bpskModulator(x,sps);
  case "QPSK"
    modulator = @(x)qpskModulator(x,sps);
  case "8PSK"
    modulator = @(x)psk8Modulator(x,sps);
  case "16QAM"
    modulator = @(x)qam16Modulator(x,sps);
  case "64QAM"
    modulator = @(x)qam64Modulator(x,sps);
  case "GFSK"
    modulator = @(x)gfskModulator(x,sps);
end
end
function y = bpskModulator(x,sps)
%bpskModulator BPSK modulator with pulse shaping
%   Y = bpskModulator(X,SPS) BPSK modulates the input X, and returns the
%   root-raised cosine pulse shaped signal Y. X must be a column vector
%   of values in the set [0 1]. The root-raised cosine filter has a
%   roll-off factor of 0.35 and spans four symbols. The output signal
%   Y has unit power.

persistent filterCoeffs
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(0.35, 4, sps);
end
% Modulate
syms = pskmod(x,2);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end

function y = qpskModulator(x,sps)
%qpskModulator QPSK modulator with pulse shaping
%   Y = qpskModulator(X,SPS) QPSK modulates the input X, and returns the
%   root-raised cosine pulse shaped signal Y. X must be a column vector
%   of values in the set [0 3]. The root-raised cosine filter has a
%   roll-off factor of 0.35 and spans four symbols. The output signal
%   Y has unit power.

persistent filterCoeffs
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(0.35, 4, sps);
end
% Modulate
syms = pskmod(x,4,pi/4);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end

function y = psk8Modulator(x,sps)
%psk8Modulator 8-PSK modulator with pulse shaping
%   Y = psk8Modulator(X,SPS) 8-PSK modulates the input X, and returns the
%   root-raised cosine pulse shaped signal Y. X must be a column vector
%   of values in the set [0 7]. The root-raised cosine filter has a
%   roll-off factor of 0.35 and spans four symbols. The output signal
%   Y has unit power.

persistent filterCoeffs
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(0.35, 4, sps);
end
% Modulate
syms = pskmod(x,8);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end

function y = qam16Modulator(x,sps)
%qam16Modulator 16-QAM modulator with pulse shaping
%   Y = qam16Modulator(X,SPS) 16-QAM modulates the input X, and returns the
%   root-raised cosine pulse shaped signal Y. X must be a column vector
%   of values in the set [0 15]. The root-raised cosine filter has a
%   roll-off factor of 0.35 and spans four symbols. The output signal
%   Y has unit power.

persistent filterCoeffs
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(0.35, 4, sps);
end
% Modulate and pulse shape
syms = qammod(x,16,'UnitAveragePower',true);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end

function y = qam64Modulator(x,sps)
%qam64Modulator 64-QAM modulator with pulse shaping
%   Y = qam64Modulator(X,SPS) 64-QAM modulates the input X, and returns the
%   root-raised cosine pulse shaped signal Y. X must be a column vector
%   of values in the set [0 63]. The root-raised cosine filter has a
%   roll-off factor of 0.35 and spans four symbols. The output signal
%   Y has unit power.

persistent filterCoeffs
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(0.35, 4, sps);
end
% Modulate
syms = qammod(x,64,'UnitAveragePower',true);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end
function y = gfskModulator(x,sps)
%gfskModulator GFSK modulator
%   Y = gfskModulator(X,SPS) GFSK modulates the input X and returns the
%   signal Y. X must be a column vector of values in the set [0 1]. The
%   BT product is 0.35 and the modulation index is 1. The output signal
%   Y has unit power.

persistent mod meanM
if isempty(mod)
  M = 2;
  mod = comm.CPMModulator(...
    'ModulationOrder', M, ...
    'FrequencyPulse', 'Gaussian', ...
    'BandwidthTimeProduct', 0.35, ...
    'ModulationIndex', 1, ...
    'SamplesPerSymbol', sps);
  meanM = mean(0:M-1);
end
% Modulate
y = mod(2*(x-meanM));
end
