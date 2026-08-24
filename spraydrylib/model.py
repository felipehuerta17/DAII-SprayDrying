from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union
import numpy as np

from spraydrylib.config import OperatingConditions, DryerParameters
from spraydrylib.properties import PhysicalProperties
from spraydrylib import physics

"""
model.py
--------
Modelo fenomenológico del Secador Spray.
Contiene:
- SprayDryer: Clase orientada a objetos para configuración, simulación y análisis.
- SimulationResult: Estructura contenedora de resultados con métodos de análisis y graficación.
- ParamsExact & simulate_time: Wrappers para 100% retrocompatibilidad con scripts existentes.
"""

# Alias para retrocompatibilidad
ParamsExact = OperatingConditions

class SimulationResult:
    """
    Contenedor de resultados de la simulación del Secador Spray.
    Permite acceso tanto como atributos de objeto como claves de diccionario (res['t'], etc.).
    """
    def __init__(self, t: np.ndarray, Ho: np.ndarray, Xo: np.ndarray, T4: np.ndarray,
                 rd: np.ndarray, params: OperatingConditions, config: DryerParameters,
                 extra: Optional[Dict[str, Any]] = None):
        self.t = np.asarray(t, dtype=float)
        self.Ho = np.asarray(Ho, dtype=float)
        self.Xo = np.asarray(Xo, dtype=float)
        self.T4 = np.asarray(T4, dtype=float)
        self.rd = np.asarray(rd, dtype=float)
        self.params = params
        self.config = config
        self.extra = extra or {}

    def __getitem__(self, key: str):
        if key == "t": return self.t
        if key == "Ho": return self.Ho
        if key == "Xo": return self.Xo
        if key == "T4": return self.T4
        if key == "rd": return self.rd
        if key == "params": return self.params
        if key == "config": return self.config
        if key in self.extra: return self.extra[key]
        raise KeyError(f"Clave desconocida: {key}")

    def __contains__(self, key: str) -> bool:
        return key in ["t", "Ho", "Xo", "T4", "rd", "params", "config"] or key in self.extra

    def keys(self):
        return ["t", "Ho", "Xo", "T4", "rd", "params", "config"] + list(self.extra.keys())

    def get_final_moisture(self, last_n_points: int = 20) -> float:
        """Calcula el contenido de humedad promedio final del sólido (Xo)."""
        n = min(last_n_points, len(self.Xo))
        return float(np.mean(self.Xo[-n:]))

    def get_energy_consumption_kw(self, cp_air: float = 1.005) -> float:
        """Calcula el gasto energético del flujo de aire caliente en kW."""
        return float(self.params.G * (self.params.T_i - self.config.T_amb) * cp_air)

    def to_dataframe(self):
        """Convierte los perfiles temporales a un DataFrame de pandas."""
        import pandas as pd
        data = {
            "Tiempo_s": self.t,
            "Humedad_aire_Ho": self.Ho,
            "Humedad_solido_Xo": self.Xo,
            "Temperatura_T4_K": self.T4,
            "Radio_gota_rd_m": self.rd
        }
        for k, v in self.extra.items():
            if isinstance(v, np.ndarray) and len(v) == len(self.t):
                data[k] = v
        return pd.DataFrame(data)

    def plot(self, save_path: Optional[str] = None):
        """Genera gráficos de la evolución temporal de estados."""
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        
        # 1. Humedad del aire
        axes[0, 0].plot(self.t, self.Ho, color="teal", lw=2)
        axes[0, 0].set_title("Humedad del aire ($H_o$)")
        axes[0, 0].set_xlabel("Tiempo (s)")
        axes[0, 0].set_ylabel("kg vapor / kg aire seco")
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Humedad del sólido
        axes[0, 1].plot(self.t, self.Xo, color="royalblue", lw=2)
        axes[0, 1].axhline(self.config.Xoc, color="gray", linestyle="--", label="Crítico $X_{oc}$")
        axes[0, 1].set_title("Contenido de humedad de la gota ($X_o$)")
        axes[0, 1].set_xlabel("Tiempo (s)")
        axes[0, 1].set_ylabel("kg agua / kg sólido seco")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Temperatura del aire
        axes[1, 0].plot(self.t, self.T4 - 273.15, color="firebrick", lw=2)
        axes[1, 0].set_title("Temperatura del aire de salida ($T_4$)")
        axes[1, 0].set_xlabel("Tiempo (s)")
        axes[1, 0].set_ylabel("Temperatura (°C)")
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Radio de la gota (primeros segundos o total)
        axes[1, 1].plot(self.t, self.rd * 1e6, color="darkgreen", lw=2)
        axes[1, 1].set_title("Radio de gota ($r_d$)")
        axes[1, 1].set_xlabel("Tiempo (s)")
        axes[1, 1].set_ylabel(r"Radio ($\mu$m)")
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()


class SprayDryer:
    """
    Modelo de Simulación Fenomenológico de Secado Spray.
    """
    def __init__(self,
                 params: Optional[OperatingConditions] = None,
                 config: Optional[DryerParameters] = None,
                 props: Optional[PhysicalProperties] = None):
        self.params = params or OperatingConditions()
        self.config = config or DryerParameters()
        self.props = props or PhysicalProperties()

    def simulate(self, tf: float = 400.0, n_steps: int = 600, method: str = "auto") -> SimulationResult:
        """
        Ejecuta la simulación temporal del secador spray.
        
        Parámetros:
        -----------
        tf : float
            Tiempo final de simulación en segundos (default 400 s).
        n_steps : int
            Número de pasos de integración (default 600).
        method : str
            Método de integración: 'auto', 'rk45', 'scipy', 'euler', o 'casadi'.
        """
        # Calcular masa seca equivalente por gota
        ms = physics.dry_mass_per_droplet(
            self.params.ri, self.params.Xi,
            self.config.A_EMP, self.config.B_EMP
        )
        
        # Estado inicial: [Ho(0), Xo(0), T4(0)]
        T4_0 = 100.0 + 273.15
        state_0 = np.array([self.params.Hi, self.params.Xi, T4_0], dtype=float)
        
        N = int(n_steps)
        t_eval = np.linspace(0.0, tf, N + 1)
        dt = float(tf) / N
        
        # Intentar solver SciPy si method es 'auto', 'scipy' o 'rk45'
        use_scipy = method in ["auto", "scipy", "rk45"]
        if use_scipy:
            try:
                from scipy.integrate import solve_ivp
                
                def rhs(t, y):
                    # Evitar valores negativos no físicos durante la integración
                    y_safe = np.array([max(y[0], 0.0), max(y[1], 1e-9), max(y[2], 200.0)], dtype=float)
                    dydt, _ = physics.compute_derivatives(t, y_safe, self.params, self.config, self.props, ms)
                    return dydt
                
                sol = solve_ivp(rhs, (0.0, tf), state_0, t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-9)
                if sol.success:
                    Ho = sol.y[0]
                    Xo = np.maximum(sol.y[1], 1e-9)
                    T4 = sol.y[2]
                    rd = np.array([
                        physics.droplet_radius_from_moisture(
                            max(x, self.config.Xoc), ms, self.config.A_EMP, self.config.B_EMP
                        ) for x in Xo
                    ])
                    return SimulationResult(sol.t, Ho, Xo, T4, rd, self.params, self.config)
            except Exception:
                pass  # Fallback a integración explícita
        
        # Integración explícita (Euler mejorado / paso adaptado)
        Ho = np.empty(N + 1)
        Xo = np.empty(N + 1)
        T4 = np.empty(N + 1)
        rd = np.empty(N + 1)
        
        Ho[0], Xo[0], T4[0] = state_0[0], state_0[1], state_0[2]
        rd[0] = physics.droplet_radius_from_moisture(self.params.Xi, ms, self.config.A_EMP, self.config.B_EMP)
        
        for k in range(N):
            current_state = np.array([Ho[k], max(Xo[k], 1e-9), T4[k]], dtype=float)
            dydt, aux = physics.compute_derivatives(t_eval[k], current_state, self.params, self.config, self.props, ms)
            
            Ho[k + 1] = max(0.0, Ho[k] + dt * dydt[0])
            Xo[k + 1] = max(1e-9, Xo[k] + dt * dydt[1])
            T4[k + 1] = T4[k] + dt * dydt[2]
            rd[k + 1] = aux["rd"]
        
        return SimulationResult(t_eval, Ho, Xo, T4, rd, self.params, self.config)


def simulate_time(params: Union[OperatingConditions, Any],
                  tf: float = 400.0,
                  n_steps: int = 600,
                  config: Optional[DryerParameters] = None) -> Dict[str, Any]:
    """
    Función de compatibilidad total con la firma histórica:
    simulate_time(params, tf=400.0, n_steps=600) -> dict con ['t', 'Ho', 'Xo', 'T4', 'rd', 'params']
    """
    if not isinstance(params, OperatingConditions):
        p = OperatingConditions(
            G=float(params.G),
            T_i=float(params.T_i),
            Hi=float(params.Hi),
            F=float(params.F),
            T_F=float(params.T_F),
            Xi=float(params.Xi),
            ri=float(params.ri)
        )
    else:
        p = params
    
    dryer = SprayDryer(params=p, config=config)
    res = dryer.simulate(tf=tf, n_steps=n_steps)
    return {
        "t": res.t,
        "Ho": res.Ho,
        "Xo": res.Xo,
        "T4": res.T4,
        "rd": res.rd,
        "params": res.params
    }
