from __future__ import annotations
import numpy as np
from math import pi
from typing import Tuple, Dict, Any, Optional, Union, Callable
from spraydrylib.config import OperatingConditions, DryerParameters
from spraydrylib.properties import PhysicalProperties

"""
physics.py
----------
Módulo central de ecuaciones fenomenológicas del secador spray.
Implementa:
1. Geometría y encogimiento de gota (masa seca m_s, radio r(X), área a_drop).
2. Transferencia de calor convectiva gas-gota (coeficiente h a partir de Nu).
3. Cinética de secado: período de tasa constante (control calor) y decreciente (difusión).
4. Pérdidas térmicas hacia el exterior por convección de pared y radiación térmica.
5. Balances dinámicos de masa y energía (sistema de ecuaciones diferenciales).
"""

def dry_mass_per_droplet(ri: float, Xi: float, A_EMP: float = 1000.0, B_EMP: float = 290.0) -> float:
    """
    Calcula la masa seca equivalente de una gota individual a partir del radio inicial ri y humedad Xi.
    
    m_s = (4 * pi * ri^3 * (A_EMP*(1+Xi) + B_EMP)) / (3 * (1+Xi)^2)
    """
    return (ri ** 3) * (4.0 * pi * (A_EMP * (1.0 + Xi) + B_EMP)) / (3.0 * (1.0 + Xi) ** 2)

def droplet_radius_from_moisture(X: float, ms: float, A_EMP: float = 1000.0, B_EMP: float = 290.0) -> float:
    """
    Calcula el radio de la gota en función de la humedad del sólido X y su masa seca ms.
    
    r(X) = ((3 * (1+X)^2 * ms) / (4 * pi * (A_EMP*(1+X) + B_EMP)))^(1/3)
    """
    # Evitar valores negativos o nulos
    X_safe = max(X, 1e-9)
    num = 3.0 * ((1.0 + X_safe) ** 2) * ms
    den = 4.0 * pi * (A_EMP * (1.0 + X_safe) + B_EMP)
    return (num / den) ** (1.0 / 3.0)

def droplet_surface_area(rd: float) -> float:
    """Área superficial de la gota: a_drop = 4 * pi * rd^2 [m^2]."""
    return 4.0 * pi * (rd ** 2)

def heat_transfer_coefficient(Nu: Union[float, Callable], k_air: float, rd: float, **kwargs) -> float:
    """
    Coeficiente de película de transferencia de calor: h = Nu * k_air / (2 * rd) [W/(m^2 K)].
    Admite Nu constante (float) o correlación personalizada (función callable).
    """
    rd_safe = max(rd, 1e-9)
    if callable(Nu):
        nu_val = float(Nu(k_air=k_air, rd=rd_safe, **kwargs))
    else:
        nu_val = float(Nu)
    return nu_val * k_air / (2.0 * rd_safe)

def wall_heat_loss_kw(T4: float, config: DryerParameters) -> float:
    """
    Pérdidas de calor a través de las paredes del secador hacia el ambiente [kW].
    
    Incluye:
    1. Convección/conducción en pared: Q_conv = U_wall * A_wall * (T4 - T_amb)
    2. Radiación térmica: Q_rad = emissivity * sigma * A_wall * (T4^4 - T_amb^4)
    """
    if config.U_wall <= 0.0 and config.emissivity <= 0.0:
        return 0.0
    
    q_conv_w = config.U_wall * config.A_wall * (T4 - config.T_amb)
    q_rad_w = 0.0
    if config.emissivity > 0.0:
        q_rad_w = config.emissivity * config.sigma_rad * config.A_wall * ((T4 ** 4) - (config.T_amb ** 4))
    
    return (q_conv_w + q_rad_w) / 1000.0  # Retorna en kW

def drying_rate_dX_dt(Xo: float, Ho: float, T4: float, rd: float, ms: float,
                      params: OperatingConditions, config: DryerParameters,
                      props: PhysicalProperties) -> float:
    """
    Calcula la derivada dXo/dt (tasa de variación de humedad en el sólido).
    
    - Si Xo >= Xoc (tasa constante, control térmico):
        dXo/dt = (h / lambda_sat) * (a_drop / ms) * (T_sat - T4)
    - Si Xo < Xoc (tasa decreciente, control difusivo):
        dXo/dt = - (4 * pi^2 / (2 * rd)^2) * De * (Xo - m_equil * Ho)
    """
    if Xo >= config.Xoc:
        # Período de tasa constante
        k_air = props.kA_air_fn(T4)
        h = heat_transfer_coefficient(config.Nu, k_air, rd)
        adrop = droplet_surface_area(rd)
        lam = props.lambda_sat_fn(config.T_sat)
        return (h / lam) * (adrop / ms) * (config.T_sat - T4)
    else:
        # Período de tasa decreciente
        De = props.De_eff_fn(config.T_sat)
        return - (4.0 * (pi ** 2) / ((2.0 * rd) ** 2)) * De * (Xo - config.m_equil * Ho)

def compute_derivatives(t: float, state: np.ndarray,
                        params: OperatingConditions,
                        config: DryerParameters,
                        props: PhysicalProperties,
                        ms: float) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Calcula el sistema completo de derivadas [dHo/dt, dXo/dt, dT4/dt] y variables algebraicas.
    
    Retorna:
    --------
    d_state : np.ndarray de forma (3,) [dHo, dXo, dT4]
    aux : dict con variables algebraicas (rd, h, Q_loss_kW, etc.)
    """
    Ho, Xo, T4 = float(state[0]), float(state[1]), float(state[2])
    
    # 1. Radio de la gota efectivo (transición suave/corte en Xoc)
    X_for_r = max(Xo, config.Xoc)
    rd = droplet_radius_from_moisture(X_for_r, ms, config.A_EMP, config.B_EMP)
    
    # 2. Propiedades termofísicas
    k_air = props.kA_air_fn(T4)
    h = heat_transfer_coefficient(config.Nu, k_air, rd)
    lam = props.lambda_sat_fn(config.T_sat)
    
    F_1 = params.F * (1.0 + params.Xi)
    F_3 = params.G * (1.0 + params.Hi)
    F_4 = params.G * (1.0 + Ho)
    
    # Humedades molares / fracciones másicas de vapor
    yv_3 = params.Hi / (1.0 - max(params.Hi, 1e-9))
    yv_4 = Ho / (1.0 - max(Ho, 1e-9))
    
    Cp_1 = props.cp_humid_air_in_fn(config.T_sat)
    Cp_3 = props.cp_humid_air_fn(params.T_i, params.Hi)
    Cp_4 = props.cp_humid_air_fn(T4, Ho)
    
    Mg = config.M_A * (1.0 + Ho)
    Cv_4 = props.cv_mixture_fn(Ho)
    
    # 3. Balances diferenciales
    # Balance de humedad del aire
    dHo_dt = (params.G / config.M_A) * (params.Hi - Ho) + (params.F / config.M_A) * (params.Xi - Xo)
    
    # Balance de humedad de la gota
    dXo_dt = drying_rate_dX_dt(Xo, Ho, T4, rd, ms, params, config, props)
    
    # Pérdidas térmicas a través de la pared (convección + radiación)
    Q_loss = wall_heat_loss_kw(T4, config)
    
    # Balance de energía en la fase gas
    # dT4/dt = (Entradas de entalpía - Salidas de entalpía + Calor latente - Perdidas) / (Mg * Cv)
    dT4_dt = (
        F_1 * Cp_1 * (params.T_F - config.T_sat)
        + F_3 * Cp_3 * (params.T_i - config.T_sat)
        - F_4 * Cp_4 * (T4 - config.T_sat)
        + lam * (F_3 * yv_3 - F_4 * yv_4)
        - Q_loss
    ) / (Mg * Cv_4)
    
    aux = {
        "rd": rd,
        "h": h,
        "Q_loss_kW": Q_loss,
        "k_air": k_air,
        "lam": lam
    }
    
    return np.array([dHo_dt, dXo_dt, dT4_dt], dtype=float), aux


def fast_simulate_final(params: OperatingConditions,
                        config: Optional[DryerParameters] = None,
                        tf: float = 400.0) -> Tuple[float, float, float]:
    """
    Integrador numérico ultrarrápido optimizado para evaluación de objetivos en optimización (MOO).
    Ejecuta en ~1 ms por evaluación sin sobrecarga de arrays intermedios.
    Retorna: (Ho_final, Xo_final, T4_final)
    """
    cfg = config or DryerParameters()
    ms = dry_mass_per_droplet(params.ri, params.Xi, cfg.A_EMP, cfg.B_EMP)
    
    Ho = float(params.Hi)
    Xo = float(params.Xi)
    T4 = 100.0 + 273.15
    
    F1 = params.F * (1.0 + params.Xi)
    F3 = params.G * (1.0 + params.Hi)
    yv3 = params.Hi / (1.0 - max(params.Hi, 1e-9))
    
    T_sat = cfg.T_sat
    cp_in_Tsat = (3774.48 + 1.15 * (T_sat - 273.15) + 3.93e-3 * (T_sat - 273.15)**2) / 1000.0
    term1 = F1 * cp_in_Tsat * (params.T_F - T_sat)
    
    A_EMP, B_EMP, Xoc = cfg.A_EMP, cfg.B_EMP, cfg.Xoc
    M_A = cfg.M_A
    m_equil = cfg.m_equil
    U_wall = cfg.U_wall
    A_wall = cfg.A_wall
    emissivity = cfg.emissivity
    sigma = cfg.sigma_rad
    T_amb = cfg.T_amb
    
    is_nu_callable = callable(cfg.Nu)
    Nu_val = 2.0 if is_nu_callable else float(cfg.Nu)
    
    # Etapa 1: Transiente inicial rápido (0 a 2s) con subpaso fino
    dt1 = 0.005
    n_steps_1 = int(2.0 / dt1)
    for _ in range(n_steps_1):
        kA = 7.4e-8 * T4 + 4.19e-6
        lambda_sat = 3180.14 - 2.508 * T_sat
        cp_in = 1.005 + 1.88 * params.Hi
        cp_out = 1.005 + 1.88 * Ho
        cv_out = 0.718 + 1.4108 * Ho
        Mg = M_A * (1.0 + Ho)
        F4 = params.G * (1.0 + Ho)
        yv4 = Ho / (1.0 - max(Ho, 1e-9))
        
        dHo = (params.G / M_A) * (params.Hi - Ho) + (params.F / M_A) * (params.Xi - Xo)
        
        if Xo >= Xoc:
            rd = ((3.0 * (1.0 + Xo)**2 * ms) / (4.0 * np.pi * (A_EMP * (1.0 + Xo) + B_EMP)))**(1.0/3.0)
            Nu_eff = cfg.Nu(kA, rd, T_gas=T4) if is_nu_callable else Nu_val
            h = (Nu_eff * kA) / (2.0 * rd)
            adrop = 4.0 * np.pi * rd**2
            dXo = (h / lambda_sat) * (adrop / ms) * (T_sat - T4)
        else:
            rd = ((3.0 * (1.0 + Xoc)**2 * ms) / (4.0 * np.pi * (A_EMP * (1.0 + Xoc) + B_EMP)))**(1.0/3.0)
            De = 1.38e-11 * T_sat - 1.55e-9
            dXo = - (4.0 * np.pi**2 / (2.0 * rd)**2) * De * (Xo - m_equil * Ho)
            
        term2 = F3 * cp_in * (params.T_i - T_sat)
        term3 = F4 * cp_out * (T4 - T_sat)
        term4 = lambda_sat * (F3 * yv3 - F4 * yv4)
        Q_loss = 0.0
        if U_wall > 0.0:
            Q_loss += U_wall * A_wall * (T4 - T_amb)
        if emissivity > 0.0:
            Q_loss += emissivity * sigma * A_wall * (T4**4 - T_amb**4)
            
        dT4 = (term1 + term2 - term3 + term4 - Q_loss/1000.0) / (Mg * cv_out)
        
        Ho += dt1 * dHo
        Xo = max(1e-9, Xo + dt1 * dXo)
        T4 += dt1 * dT4

    # Etapa 2: Secado lento y estabilización térmica (2s a tf)
    dt2 = 0.5
    n_steps_2 = int(max(tf - 2.0, 0.0) / dt2)
    for _ in range(n_steps_2):
        cp_out = 1.005 + 1.88 * Ho
        cv_out = 0.718 + 1.4108 * Ho
        Mg = M_A * (1.0 + Ho)
        F4 = params.G * (1.0 + Ho)
        yv4 = Ho / (1.0 - max(Ho, 1e-9))
        
        dHo = (params.G / M_A) * (params.Hi - Ho) + (params.F / M_A) * (params.Xi - Xo)
        
        if Xo >= Xoc:
            kA = 7.4e-8 * T4 + 4.19e-6
            lambda_sat = 3180.14 - 2.508 * T_sat
            rd = ((3.0 * (1.0 + Xo)**2 * ms) / (4.0 * np.pi * (A_EMP * (1.0 + Xo) + B_EMP)))**(1.0/3.0)
            Nu_eff = cfg.Nu(kA, rd, T_gas=T4) if is_nu_callable else Nu_val
            h = (Nu_eff * kA) / (2.0 * rd)
            adrop = 4.0 * np.pi * rd**2
            dXo = (h / lambda_sat) * (adrop / ms) * (T_sat - T4)
        else:
            rd = ((3.0 * (1.0 + Xoc)**2 * ms) / (4.0 * np.pi * (A_EMP * (1.0 + Xoc) + B_EMP)))**(1.0/3.0)
            De = 1.38e-11 * T_sat - 1.55e-9
            dXo = - (4.0 * np.pi**2 / (2.0 * rd)**2) * De * (Xo - m_equil * Ho)
            
        term2 = F3 * cp_in * (params.T_i - T_sat)
        term3 = F4 * cp_out * (T4 - T_sat)
        term4 = lambda_sat * (F3 * yv3 - F4 * yv4)
        Q_loss = 0.0
        if U_wall > 0.0:
            Q_loss += U_wall * A_wall * (T4 - T_amb)
        if emissivity > 0.0:
            Q_loss += emissivity * sigma * A_wall * (T4**4 - T_amb**4)
            
        dT4 = (term1 + term2 - term3 + term4 - Q_loss/1000.0) / (Mg * cv_out)
        
        Ho += dt2 * dHo
        Xo = max(1e-9, Xo + dt2 * dXo)
        T4 += dt2 * dT4
        
    return Ho, Xo, T4
