import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Piecewise 

# Signal Definitions
# Define a function to generate a unit impulse signal which is 1 at n=0 and 0 elsewhere
def unit_impulse(n):
    return np.array([1 if i == 0 else 0 for i in range(n)])

# Define a function to generate a unit step signal which is 1 for all n≥0
def unit_step(n):
    return np.ones(n)

# Set the signal length and generate input signals
n = 20
impulse = unit_impulse(n)  # Unit impulse signal
step = unit_step(n)        # Unit step signal

# Plot Input Signals
# Create a figure to visualize the unit impulse and unit step signals side by side
plt.figure(figsize=(12, 3))
plt.subplot(1, 2, 1)
plt.stem(impulse)
plt.title("Unit Impulse \u03b4[n]")  # Title with Unicode delta symbol
plt.grid()

plt.subplot(1, 2, 2)
plt.stem(step)
plt.title("Unit Step u[n]")  # Title with unit step notation
plt.grid()
plt.tight_layout()
plt.show()  # Display

# System Definitions
# Define the impulse response for FIR system: 3-point moving average filter
h1 = np.array([1/3, 1/3, 1/3])  # Coefficients for averaging three samples

# Define the impulse response for IIR system: decaying exponential with parameter a=0.8
a = 0.8
h2 = np.array([a**i for i in range(n)])  # Exponential decay: h[n] = a^n for n≥0

# Custom Convolution Function
# Implement a custom convolution function to compute the output of a system
def custom_convolve(x, h):
    N = len(x)      # Length of input signal
    M = len(h)      # Length of impulse response
    y = np.zeros(N + M - 1)  # Output length is N+M-1
    # Loop over each output sample
    for n in range(len(y)):
        # Sum the product of input and impulse response for valid indices
        for k in range(M):
            if 0 <= n - k < N:
                y[n] += h[k] * x[n - k]
    return y

# Response Computation
# Compute the system responses for FIR and IIR systems using custom convolution
y1_impulse_custom = custom_convolve(impulse, h1)  # FIR response to unit impulse
y1_step_custom = custom_convolve(step, h1)        # FIR response to unit step
y2_impulse_custom = custom_convolve(impulse, h2)  # IIR response to unit impulse
y2_step_custom = custom_convolve(step, h2)        # IIR response to unit step

# Validation with NumPy
# Verify custom convolution results using NumPy's built-in convolution function
y1_impulse_np = np.convolve(impulse, h1)  # NumPy convolution for FIR impulse response
y1_step_np = np.convolve(step, h1)        # NumPy convolution for FIR step response
y2_impulse_np = np.convolve(impulse, h2)  # NumPy convolution for IIR impulse response
y2_step_np = np.convolve(step, h2)        # NumPy convolution for IIR step response

# Plot Responses
# Define a helper function to plot system responses with consistent formatting
def plot_response(signal, title):
    plt.stem(signal)
    plt.title(title)
    plt.grid()

# Create a figure to display all system responses
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plot_response(y1_impulse_custom, "FIR System: Impulse Response")  # Plot FIR impulse response

plt.subplot(2, 2, 2)
plot_response(y1_step_custom, "FIR System: Step Response")       # Plot FIR step response

plt.subplot(2, 2, 3)
plot_response(y2_impulse_custom, "IIR System: Impulse Response") # Plot IIR impulse response

plt.subplot(2, 2, 4)
plot_response(y2_step_custom, "IIR System: Step Response")       # Plot IIR step response

plt.tight_layout()
plt.show()  # Display the system response plots

# System Analysis
# Define a function to analyze system properties: causality, stability, and memory
def analyze_system(h, system_name):
    is_causal = np.all(np.arange(len(h)) >= 0)       # Check if system is causal (h[n]=0 for n<0)
    is_stable = np.sum(np.abs(h)) < np.inf           # Check if system is stable (sum of |h[n]| finite)
    has_memory = len(h) > 1                          # Check if system has memory (length > 1)

    print(f"\nSystem: {system_name}")
    print(f"Causal: {'Yes' if is_causal else 'No'}")
    print(f"Stable: {'Yes' if is_stable else 'No'}")
    print(f"Has Memory: {'Yes' if has_memory else 'No'}")

# Analyze the FIR and IIR systems
analyze_system(h1, "FIR (Moving Average)")
analyze_system(h2, "IIR (Decaying Exponential)")

# Optional: Symbolic h[n]
# Define the symbolic impulse response for the IIR system using SymPy
n_sym = symbols('n', integer=True)
h_expr = a**n_sym * Piecewise((1, n_sym >= 0), (0, True))  # h[n] = a^n for n≥0, else 0
print(f"\nSymbolic h[n] for IIR system: {h_expr}")