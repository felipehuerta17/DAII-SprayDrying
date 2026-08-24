from __future__ import annotations
from typing import Tuple, Callable, Any, Optional

"""
backends.py
-----------
Selecciona dinámicamente el motor de simulación y retorna:
(simulate_fn, backend_name)

simulate_fn(params, tf, n_steps, config) -> dict con:
- t   [s]  : tiempo
- Ho  [-]  : humedad específica (kg_vapor/kg_aire_seco)
- Xo  [-]  : humedad del sólido (kg_agua/kg_sólido_seco)
- T4  [K]  : temperatura del gas en la cámara
- rd  [m]  : radio de gota
- params   : objeto de parámetros
backend_name: "idas" (CasADi/IDAS), "scipy" (SciPy RK45), o "numpy" (Euler fallback)
"""

def _has(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:
        return False

def get_simulator(prefer: Optional[str] = None) -> Tuple[Callable, str]:
    """
    Retorna la función de simulación adecuada según disponibilidad de librerías.
    
    Parámetros:
    -----------
    prefer : str, opcional
        'casadi', 'idas', 'scipy', 'numpy' o None (automático).
    """
    if prefer in ["casadi", "idas"] and _has("casadi"):
        from spraydrylib.model_casadi import simulate_time_casadi
        return simulate_time_casadi, "idas"
    
    if prefer in ["scipy", "numpy", "euler"]:
        from spraydrylib.model import simulate_time
        return simulate_time, "scipy" if _has("scipy") else "numpy"

    # Detección automática por defecto:
    if _has("casadi"):
        from spraydrylib.model_casadi import simulate_time_casadi
        return simulate_time_casadi, "idas"
    
    from spraydrylib.model import simulate_time
    return simulate_time, "scipy" if _has("scipy") else "numpy"
