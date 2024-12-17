import mpmath as mp
import numpy as np
from fractions import Fraction
from numpy.polynomial.chebyshev import chebval

mp.mp.prec = 200

T = 50
N = 69
lambda_reg = 1e-6

# Hardcoded coefficients from previous code
coeffs_fraction = [
Fraction(44839482818027,1250000000000000),
Fraction(175794849923809,500000000000000000000000000),
Fraction(-353309139803679,5000000000000000),
Fraction(70259061541811,200000000000000000000000000),
Fraction(42188755301053,625000000000000),
Fraction(175341037875829,500000000000000000000000000),
Fraction(-156269979600583,2500000000000000),
Fraction(349804711034179,1000000000000000000000000000),
Fraction(2189822785959,39062500000000),
Fraction(174294380165017,500000000000000000000000000),
Fraction(-486321876484589,10000000000000000),
Fraction(347056530927213,1000000000000000000000000000),
Fraction(407474834508307,10000000000000000),
Fraction(345217138405173,1000000000000000000000000000),
Fraction(-164581346136833,5000000000000000),
Fraction(343069057610061,1000000000000000000000000000),
Fraction(7995441335933,312500000000000),
Fraction(170282382765309,500000000000000000000000000),
Fraction(-190947780821751,10000000000000000),
Fraction(337692463517819,1000000000000000000000000000),
Fraction(136520703515377,10000000000000000),
Fraction(167261488977647,500000000000000000000000000),
Fraction(-932922375082999,100000000000000000),
Fraction(33105925105731,100000000000000000000000000),
Fraction(18998414645963,3125000000000000),
Fraction(327196196638729,1000000000000000000000000000),
Fraction(-376980133693463,100000000000000000),
Fraction(161483221965903,500000000000000000000000000),
Fraction(221987365933929,100000000000000000),
Fraction(318360007289763,1000000000000000000000000000),
Fraction(-15489602186621,12500000000000000),
Fraction(313337739420893,1000000000000000000000000000),
Fraction(327382538030181,500000000000000000),
Fraction(76967235051977,250000000000000000000000000),
Fraction(-327113105263803,1000000000000000000),
Fraction(302003511413109,1000000000000000000000000000),
Fraction(154390579085367,1000000000000000000),
Fraction(147819275389359,500000000000000000000000000),
Fraction(-344052311495971,5000000000000000000),
Fraction(144387695836649,500000000000000000000000000),
Fraction(2316475427441,80000000000000000),
Fraction(281358466261471,1000000000000000000000000000),
Fraction(-28765586130381,2500000000000000000),
Fraction(1093363555117,4000000000000000000000000),
Fraction(431928434433063,100000000000000000000),
Fraction(264723665669341,1000000000000000000000000000),
Fraction(-153261274859131,100000000000000000000),
Fraction(255391924235121,1000000000000000000000000000),
Fraction(16076092386049,31250000000000000000),
Fraction(49060946896737,200000000000000000000000000),
Fraction(-6539527194917,40000000000000000000),
Fraction(46865983653301,200000000000000000000000000),
Fraction(492442503659647,10000000000000000000000),
Fraction(69489521271,312500000000000000000000),
Fraction(-17588592336521,1250000000000000000000),
Fraction(52303762124277,250000000000000000000000000),
Fraction(47756042467999,12500000000000000000000),
Fraction(194640578213451,1000000000000000000000000000),
Fraction(-984828412119599,1000000000000000000000000),
Fraction(89157770009571,500000000000000000000000000),
Fraction(243044170069451,1000000000000000000000000),
Fraction(159699048944123,1000000000000000000000000000),
Fraction(-139912779210141,2500000000000000000000000),
Fraction(137902820769193,1000000000000000000000000000),
Fraction(1664014046649,125000000000000000000000),
Fraction(27786453807711,250000000000000000000000000),
Fraction(-5460964405603,2500000000000000000000000),
Fraction(373159298242769,5000000000000000000000000000),
Fraction(230122846951407,250000000000000000000000000)
]

coeffs_float = [float(cf) for cf in coeffs_fraction]

def approx_xi(t):
    u = t/T
    val = chebval(u, coeffs_float)
    return val

def find_zeros_fast(interval=(-50,50), coarse_step=1.0):
    zeros = []
    t_min, t_max = interval
    t = t_min
    prev_val = approx_xi(t)
    t += coarse_step
    while t <= t_max:
        val = approx_xi(t)
        if prev_val*val < 0:
            zero_t = mp.findroot(approx_xi, [t - coarse_step, t])
            zeros.append(zero_t)
        prev_val = val
        t += coarse_step
    return zeros

def known_zeros_in_range(interval=(-50,50)):
    t_min, t_max = interval
    n = 1
    known = []
    # mp.zetazero(n) returns the n-th zero on the critical line as s = 1/2 + i*gamma_n
    # Extract gamma_n by taking the imaginary part: mp.im(z)
    while True:
        z = mp.zetazero(n)
        gamma_n = mp.im(z)  # Imag part of the zero
        if gamma_n > t_max:
            break
        if gamma_n >= t_min:
            known.append(gamma_n)
        n += 1
        # Safety check if no zeros found for a long range
        if gamma_n > t_max + 100:
            break
    return known


if __name__ == "__main__":
    # Example usage:
    interval = (0,200)
    # Find approximate zeros from polynomial approximation
    zeros_found = find_zeros_fast(interval=interval, coarse_step=1.0)

    # Obtain known zeros from mpmath
    known = known_zeros_in_range(interval=interval)

    print("Approximate zeros found:", zeros_found)
    print("Known zeros in range:", known)

    # Match each approximate zero to the closest known zero and print differences
    for approx_z in zeros_found:
        # Find the closest known zero
        diffs = [abs(approx_z - kz) for kz in known]
        if diffs:
            min_diff = min(diffs)
            closest_known = known[diffs.index(min_diff)]
            print(f"Approx Zero: {approx_z}, Closest Known: {closest_known}, Diff: {min_diff}")
        else:
            print(f"Approx Zero: {approx_z}, No known zeros found in range to compare.")