import os
os.environ['MPLBACKEND'] = 'Agg' # DO THIS FIRST

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def generate_violation_heatmap(save_path="results/violation_heatmap.png"):
    """
    Generates a heatmap plotting Heart Rate vs. Systolic Blood Pressure.
    Colors areas where the standard model breaks the Shock Index medical constraint.
    """
    hr_range = np.linspace(20, 200, 100)
    sbp_range = np.linspace(40, 250, 100)
    
    HR, SBP = np.meshgrid(hr_range, sbp_range)
    
    SI = HR / SBP
    
    violation_zone = (SI < 0.3) | (SI > 2.0)
    
    plt.figure(figsize=(8, 6))
    
    cmap = plt.cm.RdYlBu_r
    cf = plt.contourf(HR, SBP, SI, levels=50, cmap=cmap, alpha=0.8)
    plt.colorbar(cf, label="Shock Index (HR / SBP)")
    
    plt.contour(HR, SBP, SI, levels=[0.3, 2.0], colors='black', linewidths=2, linestyles='--')
    
    plt.text(150, 60, 'Critical\nViolation\n(SI > 2.0)', color='white', weight='bold', fontsize=12, ha='center')
    plt.text(50, 200, 'Violation\n(SI < 0.3)', color='black', weight='bold', fontsize=12, ha='center')
    plt.text(100, 150, 'Medically\nPlausible', color='black', weight='bold', fontsize=12, ha='center')
    
    plt.xlabel('Heart Rate (bpm)', fontsize=12)
    plt.ylabel('Systolic Blood Pressure (mmHg)', fontsize=12)
    plt.title('Constraint Violation Heatmap: Shock Index', fontsize=14)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"Heatmap successfully saved to {save_path}")

if __name__ == "__main__":
    logging.info("Generating Constraint Violation Heatmap...")
    generate_violation_heatmap()
