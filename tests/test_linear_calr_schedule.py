"""Full-horizon analytic verification; these are not applied-update records."""
import math
import pytest
from src.training.schedules import complexity_log_exponents, warmup_polynomial_learning_rate as rate

COUNTS = dict(zip(('g250','g500','g750','g1000'), (377408,426560,475712,524864)))

@pytest.mark.parametrize('policy', ['uniform', 'complexity_log'])
def test_all_widths_full_horizon(policy):
    exponents = complexity_log_exponents(COUNTS) if policy == 'complexity_log' else dict.fromkeys(COUNTS, 1.)
    assert complexity_log_exponents(COUNTS) == pytest.approx(dict(zip(COUNTS, (2.,1.4432006280490555,.9471931630317243,.5))))
    for width, gamma in exponents.items():
        def lr(p): return rate(p, peak=.008, warmup_steps=64, horizon=348528, exponent=gamma)
        for p, expected in [(0,0.),(63,.007875),(64,.008),(65,.008*(348463/348464)**gamma),(174296,.008*.5**gamma),(348527,.008*(1/348464)**gamma),(348528,0.)]:
            assert lr(p) == pytest.approx(expected, rel=1e-14, abs=1e-20)
        previous = .008
        for p in range(64,348529):
            value = lr(p)
            assert math.isfinite(value) and 0 <= value <= previous
            previous = value
        assert lr(348528) == 0.

@pytest.mark.parametrize('kwargs', [dict(position=-1),dict(position=348529),dict(position=True),dict(position=1.5),dict(warmup_steps=0),dict(horizon=64),dict(exponent=0),dict(exponent=float('nan')),dict(peak=-1)])
def test_invalid_rates(kwargs):
    args=dict(position=0,peak=.008,warmup_steps=64,horizon=348528,exponent=1.); args.update(kwargs)
    with pytest.raises(ValueError): rate(**args)

@pytest.mark.parametrize('counts', [{}, {'a':0,'b':1}, {'a':1,'b':1}, {'a':True,'b':2}])
def test_invalid_complexity(counts):
    with pytest.raises(ValueError): complexity_log_exponents(counts)
