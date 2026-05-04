import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Number of nodes (QPU groups)
nodes = np.array([1, 4, 9, 16])
# Total CNOT counts measured in the simulation
cnot_counts = np.array([624.2, 676.0, 728.0, 780.0])

# Fit a 2nd-degree polynomial to the data
coeffs = np.polyfit(nodes, cnot_counts, 2)
poly = np.poly1d(coeffs)

x_line = np.linspace(nodes.min(), nodes.max(), 100)
y_line = poly(x_line)

plt.figure(figsize=(10, 6))

plt.scatter(nodes, cnot_counts, color='red', label='Simulation Data', zorder=5)
plt.plot(x_line, y_line, "b--", label='Trend (2nd-degree polynomial)')

plt.title('Scaling of CNOT count vs number of QPUs (Surface Code 13x13)', fontsize=14)
plt.xlabel('Number of QPUs (Nodes)', fontsize=12)
plt.ylabel('Total number of CNOT gates', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

plt.xscale('log', base=2)

plt.savefig('cnot_qpu_plot.png', dpi=300)  # High resolution for thesis
print('Plot saved successfully to cnot_qpu_plot.png')
