from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
from math import pi

from spraydrylib.config import OperatingConditions, DryerParameters
from spraydrylib.properties import kA_air, De_eff, lambda_sat, cp_humid_air_in, cp_humid_air, cv_mixture

"""
model_casadi.py
---------------
Formulación DAE del secador spray y su integración con IDAS vía CasADi.
Admite configuración de pérdidas de calor en pared y radiación térmica.
Estados diferenciales: x = [Ho, Xo, T4]
Estado algebraico:     z = [rd]
Restricción:          rd - r(X) = 0
"""

def simulate_time_casadi(params: Any,
                         tf: float = 400.0,
                         n_steps: int = 600,
                         config: Optional[DryerParameters] = None) -> Dict[str, Any]:
    """
    Simulación mediante CasADi e integrador IDAS.
    """
    import casadi as ca

    cfg = config or DryerParameters()
    A_EMP, B_EMP = cfg.A_EMP, cfg.B_EMP
    Nu, m_equil, M_A = cfg.Nu, cfg.m_equil, cfg.M_A
    T_sat, Xoc = cfg.T_sat, cfg.Xoc
    T_amb = cfg.T_amb
    U_wall = cfg.U_wall
    emissivity = cfg.emissivity
    sigma_rad = cfg.sigma_rad
    A_wall = cfg.A_wall

    # Símbolos
    Ho = ca.SX.sym("Ho")
    Xo = ca.SX.sym("Xo")
    T4 = ca.SX.sym("T4")
    rd = ca.SX.sym("rd")
    
    x = ca.vertcat(Ho, Xo, T4)
    z = ca.vertcat(rd)

    # Parámetros (SI)
    G   = float(params.G)
    T_i = float(params.T_i)
    Hi  = float(params.Hi)
    F   = float(params.F)
    T_F = float(params.T_F)
    Xi  = float(params.Xi)
    ri  = float(params.ri)

    # Masa seca equivalente y relación r(X)
    ms = (ri ** 3) * (4.0 * pi * (A_EMP * (1.0 + Xi) + B_EMP)) / (3.0 * (1.0 + Xi) ** 2)
    def radius_from_X(X):
        return ((3.0 * (1.0 + X) ** 2 * ms) / (4.0 * pi * (A_EMP * (1.0 + X) + B_EMP))) ** (1.0 / 3.0)

    # Propiedades (usando funciones de properties.py o expresiones simbólicas)
    def _kA_air(T): return 7.4e-8 * T + 4.19e-6
    def _De_eff(T): return 1.38e-11 * T - 1.55e-9
    def _lambda_sat(T): return 3180.14 - 2.508 * T
    def _cp_humid_air_in(T): return (3774.48 + 1.15 * (T - 273.15) + 3.93e-3 * (T - 273.15) ** 2) / 1000.0
    def _cp_humid_air(T, H): return 1.005 + 1.88 * H
    def _cv_mixture(H): return 0.718 + 1.4108 * H

    # Ecuación algebraica: rd = r(max(Xo, Xoc))
    rd_target = ca.if_else(Xo >= Xoc, radius_from_X(Xo), radius_from_X(Xoc))
    alg = rd - rd_target

    # Términos auxiliares
    adrop = 4.0 * pi * (rd ** 2)
    h = Nu * _kA_air(T4) / (2.0 * rd)
    lam = _lambda_sat(T_sat)
    De = _De_eff(T_sat)

    F_1 = F * (1.0 + Xi)
    F_3 = G * (1.0 + Hi)
    F_4 = G * (1.0 + Ho)
    yv_3 = Hi / (1.0 - Hi)
    yv_4 = Ho / (1.0 - Ho)
    Cp_1 = _cp_humid_air_in(T_sat)
    Cp_3 = _cp_humid_air(T_i, Hi)
    Cp_4 = _cp_humid_air(T4, Ho)
    Mg   = M_A * (1.0 + Ho)
    Cv_4 = _cv_mixture(Ho)

    # Pérdidas de calor en pared y radiación
    q_wall = 0.0
    if U_wall > 0.0:
        q_wall = q_wall + (U_wall * A_wall * (T4 - T_amb)) / 1000.0
    if emissivity > 0.0:
        q_wall = q_wall + (emissivity * sigma_rad * A_wall * ((T4 ** 4) - (T_amb ** 4))) / 1000.0

    # DAE
    dHo_dt = (G / M_A) * (Hi - Ho) + (F / M_A) * (Xi - Xo)
    dXo_dt = ca.if_else(
        Xo >= Xoc,
        (h / lam) * (adrop / ms) * (T_sat - T4),
        (- (4.0 * (pi ** 2)) / ((2.0 * rd) ** 2)) * De * (Xo - m_equil * Ho)
    )
    dT4_dt = (
        F_1 * Cp_1 * (T_F - T_sat)
        + F_3 * Cp_3 * (T_i - T_sat)
        - F_4 * Cp_4 * (T4 - T_sat)
        + lam * (F_3 * yv_3 - F_4 * yv_4)
        - q_wall
    ) / (Mg * Cv_4)

    ode = ca.vertcat(dHo_dt, dXo_dt, dT4_dt)
    dae = {"x": x, "z": z, "ode": ode, "alg": alg}

    # Integrador
    N = int(n_steps)
    dt = float(tf) / N
    Fint = ca.integrator("F", "idas", dae, 0.0, dt, {"abstol": 1e-9, "reltol": 1e-6})

    # Marcha temporal
    t_grid = np.linspace(0.0, tf, N + 1)
    xk = np.array([Hi, Xi, 100.0 + 273.15], dtype=float)
    zk = float(radius_from_X(Xi))

    Ho_hist = [xk[0]]
    Xo_hist = [xk[1]]
    T4_hist = [xk[2]]
    rd_hist = [zk]

    for _ in range(N):
        sol = Fint(x0=xk, z0=zk, p=[])
        xk = np.array(sol["xf"]).astype(float).squeeze()
        zf = np.array(sol["zf"]).astype(float).squeeze()
        zk = float(zf) if zf.ndim == 0 else float(zf.reshape(-1)[0])
        Ho_hist.append(float(xk[0]))
        Xo_hist.append(float(xk[1]))
        T4_hist.append(float(xk[2]))
        rd_hist.append(float(zk))

    return {
        "t": t_grid,
        "Ho": np.array(Ho_hist),
        "Xo": np.array(Xo_hist),
        "T4": np.array(T4_hist),
        "rd": np.array(rd_hist),
        "params": params
    }
