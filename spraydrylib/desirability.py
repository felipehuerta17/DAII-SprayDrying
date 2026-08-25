from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Optional, Any
import numpy as np

from spraydrylib.backends import get_simulator
from spraydrylib.config import OperatingConditions, DryerParameters
from spraydrylib import model as mdl

"""
desirability.py
---------------
Optimización multiobjetivo mediante el Enfoque de Función de Deseabilidad (DFA).
Implementa transformaciones de Harrington / Derringer & Suich y agregación geométrica.
"""

def desirability_min(f: float, L: float, U: float, s: float = 1.0) -> float:
    """
    Función de deseabilidad para objetivo a MINIMIZAR:
    - d = 1 si f <= L (óptimo deseado)
    - d = ((U - f) / (U - L))^s si L < f < U
    - d = 0 si f >= U (inaceptable)
    """
    if np.isnan(f) or np.isinf(f):
        return 0.0
    if U <= L:
        return float(f <= L)
    if f <= L:
        return 1.0
    if f >= U:
        return 0.0
    return float(((U - f) / (U - L)) ** s)

def desirability_max(f: float, L: float, U: float, s: float = 1.0) -> float:
    """
    Función de deseabilidad para objetivo a MAXIMIZAR:
    - d = 0 si f <= L
    - d = ((f - L) / (U - L))^s si L < f < U
    - d = 1 si f >= U
    """
    if np.isnan(f) or np.isinf(f):
        return 0.0
    if U <= L:
        return float(f >= U)
    if f <= L:
        return 0.0
    if f >= U:
        return 1.0
    return float(((f - L) / (U - L)) ** s)

def overall_desirability(d_list: Sequence[float], weights: Sequence[float]) -> float:
    """
    Deseabilidad global compuesta (media geométrica ponderada):
    D = exp( sum(w_i * ln(d_i)) / sum(w_i) )
    """
    d = np.clip(np.asarray(d_list, float), 1e-12, 1.0)
    w = np.asarray(weights, float)
    sw = float(np.sum(w))
    w_norm = np.ones_like(d) / len(d) if sw <= 0 else w / sw
    return float(np.exp(np.sum(w_norm * np.log(d))))

@dataclass
class Bounds:
    """Límites inferior (lb) y superior (ub) para variables de decisión."""
    lb: np.ndarray
    ub: np.ndarray

def clip_bounds(X: np.ndarray, b: Bounds) -> np.ndarray:
    return np.clip(X, b.lb, b.ub)

def build_obj2(tf: float = 400.0,
               n_steps: int = 600,
               base_params: Optional[OperatingConditions] = None,
               config: Optional[DryerParameters] = None):
    """
    Construye la función evaluadora (obj2, backend) para x = [G_kg_h, rd_m].
    """
    simulate, backend = get_simulator()
    cfg = config or DryerParameters()
    
    def obj2(x):
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
            
        sol = simulate(p, tf=float(tf), n_steps=int(n_steps), config=cfg)
        Xo = np.asarray(sol["Xo"], float)
        f1 = G_kg_s * (p.T_i - cfg.T_amb) * 1.005
        f2 = float(np.mean(Xo[-20:-1]))
        
        if not np.isfinite(f1) or not np.isfinite(f2):
            f1, f2 = 1e12, 1e12
        return np.array([f1, f2], float)
        
    return obj2, backend

def estimate_LU(samples_F: np.ndarray, pad_ratio: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estima automáticamente los límites L (deseable) y U (inaceptable)
    mediante percentiles 5% y 95% de una muestra de objetivos F.
    """
    samples_F = np.asarray(samples_F, float)
    Ls = np.percentile(samples_F, 5, axis=0)
    Us = np.percentile(samples_F, 95, axis=0)
    span = np.maximum(Us - Ls, 1e-6)
    Ls = Ls - pad_ratio * span
    Us = Us + pad_ratio * span
    return Ls, Us

def dfa_optimize(weights: Tuple[float, float],
                 bounds: Bounds,
                 Ls: Tuple[float, float],
                 Us: Tuple[float, float],
                 tf: float = 400.0,
                 n_steps: int = 600,
                 pop_size: int = 40,
                 iters: int = 20,
                 seed: int = 1,
                 base_params: Optional[OperatingConditions] = None,
                 config: Optional[DryerParameters] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Optimización evolutiva / poblacional para maximizar la deseabilidad compuesta D(x).
    
    Retorna:
    --------
    x_opt : np.ndarray
        Variables de decisión óptimas [G_kg_h, rd_m]
    f_opt : np.ndarray
        Objetivos óptimos [Energia_kW, Xo_final]
    D_opt : float
        Deseabilidad global máxima alcanzada [0 a 1]
    """
    rng = np.random.default_rng(seed)
    obj2, _ = build_obj2(tf=tf, n_steps=n_steps, base_params=base_params, config=config)
    
    D_dim = len(bounds.lb)
    X = rng.uniform(bounds.lb, bounds.ub, size=(pop_size, D_dim))

    def D_of_f(f):
        d1 = desirability_min(f[0], Ls[0], Us[0])
        d2 = desirability_min(f[1], Ls[1], Us[1])
        return overall_desirability([d1, d2], weights)

    F = np.array([obj2(x) for x in X], float)
    Ds = np.array([D_of_f(f) for f in F], float)
    sigma = 0.10 * (bounds.ub - bounds.lb)

    for _ in range(int(iters)):
        elite_count = max(2, len(Ds) // 4)
        elite_idx = np.argsort(-Ds)[:elite_count]
        elite = X[elite_idx]
        
        kids = elite[rng.integers(0, len(elite), size=len(X))] + rng.normal(0.0, 1.0, size=X.shape) * sigma
        kids = clip_bounds(kids, bounds)
        
        Fk = np.array([obj2(x) for x in kids], float)
        Dk = np.array([D_of_f(f) for f in Fk], float)
        
        X_all = np.vstack([X, kids])
        F_all = np.vstack([F, Fk])
        D_all = np.hstack([Ds, Dk])
        
        order = np.argsort(-D_all)[:len(X)]
        X, F, Ds = X_all[order], F_all[order], D_all[order]
        sigma *= 0.85

    b = int(np.argmax(Ds))
    return X[b], F[b], float(Ds[b])

def dfa_front(weights_list: Iterable[Tuple[float, float]],
              bounds: Bounds,
              Ls: Tuple[float, float],
              Us: Tuple[float, float],
              tf: float = 400.0,
              n_steps: int = 600,
              pop_size: int = 40,
              iters: int = 20,
              seed: int = 1,
              base_params: Optional[OperatingConditions] = None,
              config: Optional[DryerParameters] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera un Frente de Pareto muestreando un conjunto de vectores de pesos w con DFA.
    
    Retorna:
    --------
    X_front : np.ndarray de forma (M, 2)
    F_front : np.ndarray de forma (M, 2)
    D_front : np.ndarray de forma (M,)
    """
    Xs: List[np.ndarray] = []
    Fs: List[np.ndarray] = []
    Ds: List[float] = []
    
    for i, w in enumerate(weights_list):
        x, f, d = dfa_optimize(
            w, bounds, Ls, Us,
            tf=tf, n_steps=n_steps,
            pop_size=pop_size, iters=iters,
            seed=seed + i,
            base_params=base_params,
            config=config
        )
        Xs.append(x)
        Fs.append(f)
        Ds.append(d)
        
    return np.vstack(Xs), np.vstack(Fs), np.array(Ds, float)
