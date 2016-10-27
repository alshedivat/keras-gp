clear all;
close all;
clc;

isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;

if isOctave
    rand('state', 42);
else
    rng(42);
end

N = 100;
D = 1;
X = rand(N, D);
y = rand(N, 1);

opt = [];
hyp.lik = log(0.1);
hyp.mean = [];
hyp.cov = log(repmat([0.5; 1.0], 1, D));
xg = covGrid('create', X, 1, 128);

lik = {@likGauss};
dlik = {@(varargin) dlikGrid(varargin{:}, opt)};
inf = {@(varargin) infGrid(varargin{:}, opt)};
mean = {@meanZero};
cov = {@covGrid, {@covSEiso}, xg};
dcov = [];

fn = @(x) lik_dlik(x, y, hyp, inf, mean, cov, dcov, lik, dlik);

d = jf_checkgrad(fn, X, 1e-5, [], 0);
assert(d(1) < 1e-2);
