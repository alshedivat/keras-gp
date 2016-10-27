clear all;
close all;
clc;

isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;

if isOctave
    rand('state', 42);
else
    rng(42);
end

opt = [];
hyp.lik = log(0.1);
hyp.mean = [];
hyp.cov = log([0.5; 1.0]);

lik = {@likGauss};
dlik = {@(varargin) dlikExact(varargin{:}, opt)};
inf = {@(varargin) infExact(varargin{:}, opt)};
mean = {@meanZero};
cov = {@covSEiso};
dcov = @dcovSEiso;

N = 100;
D = 10;
X = rand(N, D);
y = rand(N, 1);

fn = @(x) lik_dlik(x, y, hyp, inf, mean, cov, dcov, lik, dlik);

d = jf_checkgrad(fn, X, 1e-5, [], 0);
assert(d(1) < 1e-6);
