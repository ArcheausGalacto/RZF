import mpmath as mp
import numpy as np

mp.mp.prec = 200

def xi(s):
    return mp.zeta(s)*0.5*s*(s-1)*mp.pi**(-s/2)*mp.gamma(s/2)

def xi_tilde(s):
    # xi_tilde(s) = 0.5*s*(s-1)*zeta(s)
    return 0.5*s*(s-1)*mp.zeta(s)

T = 50
num_points = 20000
t_values = [i*(2*T/num_points)-T for i in range(num_points)]

# Sample xi_tilde on the line s=1/2+it
xi_tilde_values = [xi_tilde(0.5 + 1j*t) for t in t_values]
xi_tilde_real = np.array([float(x.real) for x in xi_tilde_values])

# Compute and print some summary statistics
print("Summary for xi_tilde(real) on [1/2 + i(-50 to 50)]:")
print(f"Min value: {xi_tilde_real.min()}")
print(f"Max value: {xi_tilde_real.max()}")
print(f"Mean value: {xi_tilde_real.mean()}")
print(f"Std deviation: {xi_tilde_real.std()}")

# Optionally, write to a file for later analysis
filename = "xi_tilde_real_data.txt"
with open(filename, "w") as f:
    f.write("# t xi_tilde_real\n")
    for t, val in zip(t_values, xi_tilde_real):
        f.write(f"{t} {val}\n")
print(f"Data written to {filename}")

# (Optional) Plot the data using matplotlib if available
# This gives a visual insight into the behavior of xi_tilde along the imaginary axis.
# You must have matplotlib installed (e.g., pip install matplotlib)
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,6))
    plt.plot(t_values, xi_tilde_real, label='Real part of xi_tilde(1/2 + it)')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("t")
    plt.ylabel("xi_tilde_real(1/2 + i t)")
    plt.title("Real part of xi_tilde along the critical line")
    plt.grid(True)
    plt.legend()
    plt.savefig("xi_tilde_plot.png")
    print("Plot saved to xi_tilde_plot.png")
except ImportError:
    print("matplotlib not installed, skipping plot.")
