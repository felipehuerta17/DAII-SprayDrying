from __future__ import annotations
from typing import Optional, Sequence, Dict, Any, List
import numpy as np

"""
plotting.py
-----------
Funciones de visualización estandarizadas y de alta calidad para:
- Perfiles dinámicos de simulación (Temperatura, Humedad, Radio de gota).
- Frentes de Pareto individuales y superpuestos.
- Puntos seleccionados por MCDM (TOPSIS y DFA).
"""

def set_academic_style():
    """Aplica configuraciones estéticas a matplotlib."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "lines.linewidth": 2,
        "grid.alpha": 0.3
    })

def plot_pareto_front(F: np.ndarray,
                      title: str = "Frente de Pareto",
                      label: str = "Frente de Pareto",
                      color: str = "#1f77b4",
                      save_path: Optional[str] = None):
    """Grafica un frente de Pareto bidimensional (Energía vs Humedad)."""
    import matplotlib.pyplot as plt
    set_academic_style()
    plt.figure(figsize=(7.5, 4.5))
    plt.scatter(F[:, 0], F[:, 1], s=36, facecolor=color, edgecolor="black", alpha=0.9, label=label)
    plt.xlabel("Gasto energético (kW)")
    plt.ylabel(r"Contenido de agua en el droplet $\left(\frac{kg_{agua}}{kg_{sólidos}}\right)$")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()

def plot_pareto_comparison(fronts_dict: Dict[str, np.ndarray],
                           title: str = "Comparación de Frentes de Pareto",
                           save_path: Optional[str] = None):
    """
    Superpone múltiples frentes de Pareto para comparación (ej. NSGA-II vs DFA).
    """
    import matplotlib.pyplot as plt
    set_academic_style()
    plt.figure(figsize=(7.5, 4.5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    
    for i, (label, F) in enumerate(fronts_dict.items()):
        c = colors[i % len(colors)]
        plt.scatter(F[:, 0], F[:, 1], s=32, facecolor=c, edgecolor="black", alpha=0.85, label=label)
        
    plt.xlabel("Gasto energético (kW)")
    plt.ylabel(r"Contenido de agua en el droplet $\left(\frac{kg_{agua}}{kg_{sólidos}}\right)$")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()

def plot_topsis_vs_dfa(F_pareto: np.ndarray,
                       y_topsis: np.ndarray,
                       y_dfa: np.ndarray,
                       weights_labels: Sequence[str],
                       title: str = "Selección por TOPSIS y DFA (mismos pesos)",
                       save_path: Optional[str] = None):
    """
    Grafica el frente base de Pareto y marca las soluciones elegidas por TOPSIS y DFA.
    """
    import matplotlib.pyplot as plt
    set_academic_style()
    plt.figure(figsize=(8.0, 5.2))
    
    # Frente base
    plt.scatter(F_pareto[:, 0], F_pareto[:, 1], s=28, color="0.70", edgecolor="gray", alpha=0.6, label="Frente NSGA-II")
    
    # Marcadores
    topsis_markers = ["*", "D", "s", "p", "h"]
    dfa_markers = ["X", "^", "P", "v", "<"]
    
    for i, (pt, w_lab) in enumerate(zip(y_topsis, weights_labels)):
        m = topsis_markers[i % len(topsis_markers)]
        plt.scatter([pt[0]], [pt[1]], s=180, marker=m, edgecolor="black", color="#1f77b4", label=f"TOPSIS {w_lab}")
        
    for i, (pt, w_lab) in enumerate(zip(y_dfa, weights_labels)):
        m = dfa_markers[i % len(dfa_markers)]
        plt.scatter([pt[0]], [pt[1]], s=180, marker=m, edgecolor="black", color="#ff7f0e", label=f"DFA {w_lab}")
        
    plt.xlabel("Gasto energético (kW)")
    plt.ylabel(r"Contenido de agua en el droplet $\left(\frac{kg_{agua}}{kg_{sólidos}}\right)$")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(ncol=2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()
