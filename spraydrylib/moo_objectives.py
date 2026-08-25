from __future__ import annotations
from typing import Optional, Union, Sequence, Any
import numpy as np

from spraydrylib.backends import get_simulator
from spraydrylib.config import OperatingConditions, DryerParameters
from spraydrylib import model as mdl

"""
moo_objectives.py
-----------------
Evaluación de objetivos del problema de optimización multiobjetivo (MOO):
f1 = Gasto energético del aire caliente (kW)  [MIN]
f2 = Contenido final de humedad del polvo (kg agua / kg sólido) [MIN]
"""

def two_objectives_vec(x: Sequence[float],
                       base_params: Optional[OperatingConditions] = None,
                       config: Optional[DryerParameters] = None,
                       tf: float = 400.0,
                       n_steps: int = 600) -> np.ndarray:
    """
    Devuelve np.array([energia_kW, humedad_final]) para variables de decisión x = [G_kg_h, rd_m].
    
    Parámetros:
    -----------
    x : Sequence[float]
        [G_kg_h, rd_m] = [Caudal de aire en kg/h, Radio inicial de gota en m]
    base_params : OperatingConditions, opcional
        Condiciones base del secador (temperaturas, flujos, humedad inicial).
    config : DryerParameters, opcional
        Parámetros del secador (pérdidas de pared, constantes).
    """
    simulate, _ = get_simulator()
    G_h, rd_i = float(x[0]), float(x[1])
    G_kg_s = G_h / 3600.0
    
    if base_params is None:
        p = mdl.OperatingConditions(
            G=G_kg_s,
            T_i=145.0 + 273.15,
            Hi=0.0152,
            F=4.5 / 3600.0,
            T_F=28.0 + 273.15,
            Xi=3.0,
            ri=rd_i
        )
    else:
        p = base_params.copy(G=G_kg_s, ri=rd_i)
        
    cfg = config or DryerParameters()
    from spraydrylib.physics import fast_simulate_final
    ho, xo, t4 = fast_simulate_final(p, config=cfg, tf=tf)
    
    f1 = G_kg_s * (p.T_i - cfg.T_amb) * 1.005
    f2 = float(xo)
    
    if not np.isfinite(f1) or not np.isfinite(f2):
        return np.array([1e9, 1e9], dtype=float)
        
    return np.array([f1, f2], dtype=float)
