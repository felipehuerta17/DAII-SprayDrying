from __future__ import annotations

"""
spraydrylib
===========
Paquete de simulación fenomenológica y optimización multiobjetivo para el secado spray de leche.
Diplomado en Automática e Informática Industrial (DAII).

Módulos:
- config: Estructuras de datos (OperatingConditions, DryerParameters).
- properties: Propiedades termofísicas y de transporte (kA, De, lambda, Cp, Cv).
- physics: Balances diferenciales, balances de energía, pérdida de calor y radiación.
- model: SprayDryer, SimulationResult, ParamsExact, simulate_time.
- model_casadi: Integrador IDAS con CasADi para formulación DAE.
- backends: Selección automática del motor de simulación.
- moo_objectives: Evaluación de funciones multiobjetivo.
- desirability: Enfoque de funciones de deseabilidad (DFA).
- pymoo_problem: Definición de problema multiobjetivo para pymoo (NSGA-II).
- mcdm: Algoritmos de decisión multicriterio (TOPSIS).
- plotting: Visualizaciones estándar.
"""

from spraydrylib.config import OperatingConditions, DryerParameters
from spraydrylib.properties import PhysicalProperties
from spraydrylib.model import SprayDryer, SimulationResult, ParamsExact, simulate_time
from spraydrylib.backends import get_simulator
from spraydrylib.moo_objectives import two_objectives_vec
from spraydrylib.desirability import (
    desirability_min,
    desirability_max,
    overall_desirability,
    Bounds,
    clip_bounds,
    build_obj2,
    estimate_LU,
    dfa_optimize,
    dfa_front
)
from spraydrylib.pymoo_problem import MOOSprayProblem, contenido_energetico_kw
from spraydrylib.mcdm import topsis, TOPSIS, build_comparison_dataframe
from spraydrylib.plotting import (
    plot_pareto_front,
    plot_pareto_comparison,
    plot_topsis_vs_dfa
)

__version__ = "2.0.0"

__all__ = [
    "OperatingConditions",
    "DryerParameters",
    "PhysicalProperties",
    "SprayDryer",
    "SimulationResult",
    "ParamsExact",
    "simulate_time",
    "get_simulator",
    "two_objectives_vec",
    "desirability_min",
    "desirability_max",
    "overall_desirability",
    "Bounds",
    "clip_bounds",
    "build_obj2",
    "estimate_LU",
    "dfa_optimize",
    "dfa_front",
    "MOOSprayProblem",
    "contenido_energetico_kw",
    "topsis",
    "TOPSIS",
    "build_comparison_dataframe",
    "plot_pareto_front",
    "plot_pareto_comparison",
    "plot_topsis_vs_dfa",
]
