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
D = 10;

opt = [];
hyp.lik = log(0.1);
hyp.mean = [];
hyp.cov = [log(0.5*ones(D, 1)); log(1.0)];

cov = {@covSEard};
dcov = @dcovSEard;

fn = @(x) cov_dcov(x, hyp, cov, dcov, opt);

X = rand(N, D);

d = jf_checkgrad(fn, X, 1e-5, [], 0);
assert(d(1) < 1e-6);
