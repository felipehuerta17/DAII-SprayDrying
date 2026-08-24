from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np

"""
properties.py
-------------
Modelos termodinámicos y de transporte para aire húmedo, vapor de agua y gota de leche.
Todas las funciones aceptan tanto escalares float como arrays de NumPy o símbolos CasADi.
"""

def kA_air(T) -> float:
    """
    Conductividad térmica del aire [W / (m * K)].
    k_A(T) = 7.4e-8 * T + 4.19e-6
    """
    return 7.4e-8 * T + 4.19e-6

def De_eff(T) -> float:
    """
    Difusividad efectiva de vapor de agua en la gota / aire [m^2 / s].
    D_e(T) = 1.38e-11 * T - 1.55e-9
    """
    return 1.38e-11 * T - 1.55e-9

def lambda_sat(T) -> float:
    """
    Calor latente de vaporización del agua [kJ / kg].
    lambda(T) = 3180.14 - 2.508 * T
    """
    return 3180.14 - 2.508 * T

def cp_humid_air_in(T) -> float:
    """
    Capacidad calorífica del aire de entrada [kJ / (kg * K)].
    """
    T_C = T - 273.15
    return (3774.48 + 1.15 * T_C + 3.93e-3 * (T_C ** 2)) / 1000.0

def cp_humid_air(T, H) -> float:
    """
    Capacidad calorífica del aire húmedo en función de la humedad específica H [kJ / (kg_aire_seco * K)].
    cp(H) = 1.005 + 1.88 * H
    """
    return 1.005 + 1.88 * H

def cv_mixture(H) -> float:
    """
    Calor específico a volumen constante de la mezcla gaseosa [kJ / (kg * K)].
    cv(H) = 0.718 + 1.4108 * H
    """
    return 0.718 + 1.4108 * H


@dataclass
class PhysicalProperties:
    """
    Contenedor de funciones de propiedades físicas para fácil personalización por estudiantes.
    Permite reemplazar cualquier propiedad por una función personalizada.
    """
    kA_air_fn: Callable = kA_air
    De_eff_fn: Callable = De_eff
    lambda_sat_fn: Callable = lambda_sat
    cp_humid_air_in_fn: Callable = cp_humid_air_in
    cp_humid_air_fn: Callable = cp_humid_air
    cv_mixture_fn: Callable = cv_mixture
