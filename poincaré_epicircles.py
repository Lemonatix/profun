# Double pendulum: Poincaré circles + Fourier “epicycles” (portable version)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.fft import rfft, rfftfreq

# ---------- Double pendulum dynamics ----------
g = 9.81
L1 = 1.0; L2 = 1.0
m1 = 1.0; m2 = 1.0

def deriv(state):
    theta1, omega1, theta2, omega2 = state
    delta = theta2 - theta1
    den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta)**2
    den2 = (L2 / L1) * den1

    domega1 = (m2 * L1 * omega1**2 * np.sin(delta) * np.cos(delta)
               + m2 * g * np.sin(theta2) * np.cos(delta)
               + m2 * L2 * omega2**2 * np.sin(delta)
               - (m1 + m2) * g * np.sin(theta1)) / den1

    domega2 = (-m2 * L2 * omega2**2 * np.sin(delta) * np.cos(delta)
               + (m1 + m2) * g * np.sin(theta1) * np.cos(delta)
               - (m1 + m2) * L1 * omega1**2 * np.sin(delta)
               - (m1 + m2) * g * np.sin(theta2)) / den2

    return np.array([omega1, domega1, omega2, domega2])

def rk4_step(state, dt):
    k1 = deriv(state)
    k2 = deriv(state + 0.5 * dt * k1)
    k3 = deriv(state + 0.5 * dt * k2)
    k4 = deriv(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# ---------- Integrate a low-energy trajectory ----------
dt = 0.002
T = 80.0
N = int(T / dt)

theta1_0, omega1_0 = 0.22, 0.0
theta2_0, omega2_0 = -0.15, 0.0
state = np.array([theta1_0, omega1_0, theta2_0, omega2_0], dtype=float)

t = np.linspace(0, T, N, endpoint=False)
th1 = np.zeros(N); om1 = np.zeros(N)
th2 = np.zeros(N); om2 = np.zeros(N)

for i in range(N):
    th1[i], om1[i], th2[i], om2[i] = state
    state = rk4_step(state, dt)

# ---------- Poincaré section: sample when theta2 crosses 0 upward ----------
theta1_pts, omega1_pts = [], []
for i in range(1, N):
    if th2[i-1] < 0.0 and th2[i] >= 0.0 and om2[i] > 0.0:
        frac = (0.0 - th2[i-1]) / (th2[i] - th2[i-1] + 1e-15)
        theta1_cross = th1[i-1] + frac * (th1[i] - th1[i-1])
        omega1_cross = om1[i-1] + frac * (om1[i] - om1[i-1])
        theta1_pts.append(theta1_cross); omega1_pts.append(omega1_cross)

theta1_pts = np.array(theta1_pts); omega1_pts = np.array(omega1_pts)

plt.figure(figsize=(6,6), dpi=130)
plt.scatter(theta1_pts, omega1_pts, s=8, alpha=0.7)
plt.xlabel(r"$\theta_1$ (rad)"); plt.ylabel(r"$\omega_1$ (rad/s)")
plt.title("Double pendulum Poincaré section\n(sample when $\\Theta_2$ crosses 0 upward)")
plt.grid(True); plt.show()

# ---------- Fourier spectrum of theta1(t) ----------
x = th1 - np.mean(th1)
Y = rfft(x); freqs = rfftfreq(N, dt)
amps = np.abs(Y) / N

plt.figure(figsize=(7,4), dpi=130)
plt.plot(freqs, amps)
plt.xlim(0, 10)
plt.xlabel("Frequency (Hz)"); plt.ylabel("Amplitude")
plt.title(r"Fourier spectrum of $\theta_1(t)$")
plt.grid(True); plt.show()

# ---------- Top Fourier peaks table (print + save CSV) ----------
min_freq = 0.05
mask = freqs > min_freq
idx = np.argsort(amps[mask])[::-1][:8]
top_freqs = freqs[mask][idx]
top_amps  = amps[mask][idx]
top_phases = np.angle(Y[mask][idx])

df = pd.DataFrame({"frequency_Hz": top_freqs,
                   "amplitude": top_amps,
                   "phase_rad": top_phases}).sort_values("frequency_Hz")
print("\nTop Fourier components (theta1):\n", df.head(10).to_string(index=False))
df.to_csv("top_fourier_components.csv", index=False)
print("\nSaved full table to top_fourier_components.csv")

# ---------- Reconstruct from top K peaks ----------
K = min(6, len(top_freqs))
recon = np.zeros_like(x)
order = np.argsort(top_amps)[-K:]
for j in order:
    f = top_freqs[j]; A = top_amps[j]; phi = top_phases[j]
    recon += 2*A * np.cos(2*np.pi*f*t + phi)  # 2× for positive-freq half

plt.figure(figsize=(7,4), dpi=130)
plt.plot(t, x, label="original (detrended)")
plt.plot(t, recon, label=f"reconstruction with top {K} peaks", linewidth=1.25)
plt.xlabel("time (s)"); plt.ylabel(r"$\theta_1$ (rad)")
plt.title(r"Epicycle view: sum of a few rotating modes ≈ $\theta_1(t)$")
plt.legend(); plt.grid(True); plt.show()

print(
    "\nNotes:\n"
    "• Closed curves in the Poincaré section = slice of a 2-torus (quasi-periodic motion).\n"
    "• FFT shows a few sharp peaks (two fundamentals + combos).\n"
    "• Summing a handful of peaks reconstructs θ1(t) → the ‘epicycle’ picture.\n"
)