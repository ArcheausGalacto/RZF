import numpy as np
import mpmath as mp
from mpmath import zetazero
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk

# Configuration for the range and zero data
M = 5          # Number of known zeros
gamma_min = 10.0
gamma_max = 40.0
num_points = 1000

mp.mp.prec = 100
gamma_values = np.array([float(zetazero(k).imag) for k in range(1, M+1)])
f_values = np.ones(M)

# Updated default parameters as requested
initial_alpha = 1.0
initial_beta_min = 0.83
initial_beta_max = 2.5
initial_N = 1350
initial_scale = 0.1

class App:
    def __init__(self, master):
        self.master = master
        master.title("Gaussian-Modulated Harmonic Approximation with Scaling")

        # Frame for the plot
        self.frame_plot = tk.Frame(master)
        self.frame_plot.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Matplotlib figure and axes
        self.fig = Figure(figsize=(8,6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_plot)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Frame for sliders
        self.frame_sliders = tk.Frame(master)
        self.frame_sliders.pack(side=tk.BOTTOM, fill=tk.X)

        # Slider for alpha
        tk.Label(self.frame_sliders, text="alpha").pack(side=tk.LEFT)
        self.scale_alpha = tk.Scale(self.frame_sliders, from_=0.1, to=5.0, resolution=0.1, orient=tk.HORIZONTAL, command=self.update_plot)
        self.scale_alpha.set(initial_alpha)
        self.scale_alpha.pack(side=tk.LEFT)

        # Slider for beta_min
        tk.Label(self.frame_sliders, text="beta_min").pack(side=tk.LEFT)
        self.scale_beta_min = tk.Scale(self.frame_sliders, from_=0.01, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, command=self.update_plot)
        self.scale_beta_min.set(initial_beta_min)
        self.scale_beta_min.pack(side=tk.LEFT)

        # Slider for beta_max
        tk.Label(self.frame_sliders, text="beta_max").pack(side=tk.LEFT)
        self.scale_beta_max = tk.Scale(self.frame_sliders, from_=0.0, to=10.0, resolution=0.1, orient=tk.HORIZONTAL, command=self.update_plot)
        self.scale_beta_max.set(initial_beta_max)
        self.scale_beta_max.pack(side=tk.LEFT)

        # Slider for N
        tk.Label(self.frame_sliders, text="N").pack(side=tk.LEFT)
        self.scale_N = tk.Scale(self.frame_sliders, from_=50, to=2000, resolution=50, orient=tk.HORIZONTAL, command=self.update_plot)
        self.scale_N.set(initial_N)
        self.scale_N.pack(side=tk.LEFT)

        # Slider for scale_factor
        tk.Label(self.frame_sliders, text="scale_factor").pack(side=tk.LEFT)
        self.scale_factor = tk.Scale(self.frame_sliders, from_=0.1, to=5.0, resolution=0.1, orient=tk.HORIZONTAL, command=self.update_plot)
        self.scale_factor.set(initial_scale)
        self.scale_factor.pack(side=tk.LEFT)

        # Draw initial plot
        self.plot_function(initial_alpha, initial_beta_min, initial_beta_max, initial_N, initial_scale)

    def plot_function(self, alpha, beta_min, beta_max, N, scale):
        # Compute frequencies
        betas = np.linspace(beta_min, beta_max, N)

        # Construct the Phi matrix for the given parameters
        M = len(gamma_values)
        Phi = np.zeros((M, N), dtype=float)
        for i, g in enumerate(gamma_values):
            g_scaled = g * scale
            exp_part = np.exp(-alpha * np.pi * g_scaled**2)
            for j, b in enumerate(betas):
                Phi[i,j] = exp_part * np.cos(2*np.pi*b*g_scaled)

        # Solve least squares
        c, residuals, rank, s = np.linalg.lstsq(Phi, f_values, rcond=None)

        def F(g):
            g_scaled = g * scale
            exp_part = np.exp(-alpha * np.pi * g_scaled**2)
            return np.sum(c * exp_part * np.cos(2*np.pi*betas*g_scaled))

        g_range = np.linspace(gamma_min, gamma_max, num_points)
        F_values = [F(g) for g in g_range]

        # Clear and plot
        self.ax.clear()
        self.ax.plot(g_range, F_values, label='Gaussian-Modulated Harmonic Approx')

        # Mark the known zeros
        for gv in gamma_values:
            self.ax.axvline(x=gv, color='r', linestyle='--', alpha=0.7)
            self.ax.text(gv, 1.05, f'γ={gv:.2f}', rotation=90, verticalalignment='bottom', color='red')

        self.ax.set_title('Gaussian-Modulated Harmonic Approximation with Scaling')
        self.ax.set_xlabel('gamma')
        self.ax.set_ylabel('F(gamma)')
        self.ax.set_ylim(0, 1.2)
        self.ax.grid(True)
        self.ax.legend()
        self.canvas.draw()

    def update_plot(self, event=None):
        alpha = float(self.scale_alpha.get())
        beta_min = float(self.scale_beta_min.get())
        beta_max = float(self.scale_beta_max.get())
        N = int(self.scale_N.get())
        scale = float(self.scale_factor.get())
        self.plot_function(alpha, beta_min, beta_max, N, scale)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
