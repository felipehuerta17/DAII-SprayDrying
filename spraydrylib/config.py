from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any
import copy

"""
config.py
---------
Estructuras de datos para condiciones operativas, parámetros de diseño del secador,
y constantes del modelo fenomenológico del Secador Spray.
Todas las unidades están en el Sistema Internacional (SI).
"""

@dataclass
class OperatingConditions:
    """
    Condiciones de operación del secador spray.
    
    Atributos:
    ----------
    G : float
        Caudal másico de aire seco de entrada [kg/s]. (ej: 650 kg/h = 650/3600 kg/s)
    T_i : float
        Temperatura del aire de secado a la entrada [K]. (ej: 145°C = 418.15 K)
    Hi : float
        Humedad específica del aire de entrada [kg vapor / kg aire seco]. (ej: 0.0152)
    F : float
        Flujo másico de sólidos secos de alimentación [kg/s]. (ej: 4.5 kg/h = 4.5/3600 kg/s)
    T_F : float
        Temperatura de la suspensión líquida / leche de alimentación [K]. (ej: 28°C = 301.15 K)
    Xi : float
        Contenido inicial de humedad de la gota [kg agua / kg sólido seco]. (ej: 3.0)
    ri : float
        Radio inicial de la gota producida por el atomizador [m]. (ej: 5.0e-5 m = 50 um)
    """
    G: float = 650.0 / 3600.0
    T_i: float = 145.0 + 273.15
    Hi: float = 0.0152
    F: float = 4.5 / 3600.0
    T_F: float = 28.0 + 273.15
    Xi: float = 3.0
    ri: float = 5.0e-5

    @property
    def G_kg_h(self) -> float:
        """Caudal másico de aire en kg/h."""
        return self.G * 3600.0

    @G_kg_h.setter
    def G_kg_h(self, val: float):
        self.G = val / 3600.0

    @property
    def F_kg_h(self) -> float:
        """Caudal másico de sólidos en kg/h."""
        return self.F * 3600.0

    @F_kg_h.setter
    def F_kg_h(self, val: float):
        self.F = val / 3600.0

    @property
    def ri_microns(self) -> float:
        """Radio de la gota en micrómetros (um)."""
        return self.ri * 1.0e6

    @ri_microns.setter
    def ri_microns(self, val: float):
        self.ri = val * 1.0e-6

    def copy(self, **kwargs) -> OperatingConditions:
        """Retorna una copia con posibles modificaciones de atributos."""
        new_inst = copy.deepcopy(self)
        for k, v in kwargs.items():
            if hasattr(new_inst, k):
                setattr(new_inst, k, v)
            else:
                raise AttributeError(f"OperatingConditions no tiene el atributo '{k}'")
        return new_inst

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> OperatingConditions:
        valid_keys = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid_keys)


@dataclass
class DryerParameters:
    """
    Parámetros físicos, geométricos y constantes empíricas del secador.
    
    Atributos:
    ----------
    M_A : float
        Masa efectiva de aire seco contenida en la cámara de secado (holdup) [kg].
    T_sat : float
        Temperatura de referencia / saturación adiabática de la gota [K].
    T_amb : float
        Temperatura ambiente exterior [K].
    Xoc : float
        Humedad crítica de transición entre tasa constante y decreciente [kg agua / kg sólido].
    Nu : float
        Número de Nusselt efectivo para transferencia de calor gas-gota [-].
    m_equil : float
        Factor de equilibrio higroscópico X-H en régimen difusivo [-].
    A_EMP : float
        Parámetro empírico A de densidad efectiva de la mezcla gota [kg/m^3].
    B_EMP : float
        Parámetro empírico B de densidad efectiva de la mezcla gota [kg/m^3].
    A_wall : float
        Área superficial de la pared del secador [m^2] (para pérdidas de calor).
    U_wall : float
        Coeficiente global de transferencia de calor en pared [W/(m^2 K)] (0 = adiabático).
    emissivity : float
        Emisividad de radiación térmica de la superficie [0 a 1] (0 = sin radiación).
    sigma_rad : float
        Constante de Stefan-Boltzmann [W/(m^2 K^4)].
    """
    M_A: float = 7.0
    T_sat: float = 43.0 + 273.15
    T_amb: float = 20.0 + 273.15
    Xoc: float = 0.25
    Nu: Any = 2.0
    m_equil: float = 1.5
    A_EMP: float = 1000.0
    B_EMP: float = 290.0
    A_wall: float = 15.0
    U_wall: float = 0.0
    emissivity: float = 0.0
    sigma_rad: float = 5.670374419e-8

    def copy(self, **kwargs) -> DryerParameters:
        new_inst = copy.deepcopy(self)
        for k, v in kwargs.items():
            if hasattr(new_inst, k):
                setattr(new_inst, k, v)
            else:
                raise AttributeError(f"DryerParameters no tiene el atributo '{k}'")
        return new_inst

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> DryerParameters:
        valid_keys = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid_keys)
