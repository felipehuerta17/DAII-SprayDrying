from __future__ import annotations
from typing import Optional, Any
import numpy as np

try:
    from pymoo.core.problem import Problem
except ImportError:
    # Dummy base class if pymoo is not yet installed
    class Problem:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

from spraydrylib.backends import get_simulator
from spraydrylib.config import OperatingConditions, DryerParameters
from spraydrylib import model as mdl

"""
pymoo_problem.py
----------------
Definición de la clase MOOSprayProblem compatible con la librería pymoo (NSGA-II).
Variables de decisión:
- x0: G_kg_h [kg/h] (caudal de aire caliente)
- x1: rd_m   [m]    (radio inicial de la gota)

Funciones objetivo (ambas a MINIMIZAR):
- f0: Contenido energético del aire caliente (kW)
- f1: Contenido de agua final promedio en el polvo (kg agua / kg sólido)
"""

def contenido_energetico_kw(G_kg_s: float, T_i_K: float, T_amb_K: float = 293.15, cp_air_kJ_kgK: float = 1.005) -> float:
    """Estimación simple de gasto energético del aire húmedo (kW)."""
    return float(G_kg_s * (T_i_K - T_amb_K) * cp_air_kJ_kgK)


class MOOSprayProblem(Problem):
    """
    Problema MOO compatible con pymoo:
    x = [G_kg_h, rd_m] -> f = [energia_kW, Xo_prom]
    """
    def __init__(self,
                 tf: float = 400.0,
                 n_steps: int = 600,
                 xl: Optional[np.ndarray] = None,
                 xu: Optional[np.ndarray] = None,
                 base_params: Optional[OperatingConditions] = None,
                 config: Optional[DryerParameters] = None):
        
        xl_bounds = xl if xl is not None else np.array([462.0, 3.5e-5], dtype=float)
        xu_bounds = xu if xu is not None else np.array([858.0, 6.5e-5], dtype=float)
        
        super().__init__(
            n_var=2,
            n_obj=2,
            n_constr=0,
            xl=xl_bounds,
            xu=xu_bounds
        )
        self.tf = float(tf)
        self.n_steps = int(n_steps)
        self.base_params = base_params or OperatingConditions()
        self.config = config or DryerParameters()
        self.simulate, self.backend = get_simulator()

    def _evaluate(self, X, out, *args, **kwargs):
        F_vals = []
        for individuo in X:
            G_h = float(individuo[0])
            rd_i = float(individuo[1])
            G_s = G_h / 3600.0
            
            p = self.base_params.copy(G=G_s, ri=rd_i)
            sol = self.simulate(p, tf=self.tf, n_steps=self.n_steps, config=self.config)
            
            Xo = sol["Xo"]
            f1 = contenido_energetico_kw(G_s, p.T_i, T_amb_K=self.config.T_amb)
            f2 = float(np.mean(Xo[-20:-1]))
            
            if not np.isfinite(f1) or not np.isfinite(f2):
                f1, f2 = 1e9, 1e9
                
            F_vals.append([f1, f2])
            
        out["F"] = np.asarray(F_vals, dtype=float)
