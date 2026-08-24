from __future__ import annotations
from typing import Sequence, List, Tuple, Optional, Dict, Any, Union
import numpy as np

"""
mcdm.py
-------
Algoritmos de Toma de Decisiones Multicriterio (MCDM).
Implementa:
- TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution).
- Generación de tablas comparativas y formateo para estudiantes.
"""

def topsis(F: np.ndarray,
           weights: Sequence[float],
           max_obj: Sequence[int] = (),
           min_obj: Sequence[int] = (0, 1)) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Algoritmo TOPSIS para selección sobre un frente de Pareto.
    
    Parámetros:
    -----------
    F : np.ndarray de forma (N, M)
        Matriz de evaluación de objetivos (N alternativas, M criterios).
    weights : Sequence[float] de longitud M
        Ponderaciones de importancia de cada criterio.
    max_obj : Sequence[int]
        Índices de columnas de objetivos a MAXIMIZAR.
    min_obj : Sequence[int]
        Índices de columnas de objetivos a MINIMIZAR.
        
    Retorna:
    --------
    best_idx : int
        Índice de la mejor alternativa seleccionada.
    scores : np.ndarray de forma (N,)
        Puntaje de similitud relativa C_i para cada alternativa (mayor es mejor).
    v_ideal : np.ndarray de forma (M,)
        Vector de la solución ideal positiva.
    v_anti : np.ndarray de forma (M,)
        Vector de la solución ideal negativa (anti-ideal).
    """
    F_mat = np.asarray(F, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = w / (w.sum() if w.sum() > 0 else 1.0)
    
    # 1. Normalización vectorial
    norm = np.linalg.norm(F_mat, axis=0)
    norm[norm == 0] = 1.0
    R = F_mat / norm
    
    # 2. Ponderación
    V = R * w
    
    # 3. Determinación de soluciones ideales
    M_crit = V.shape[1]
    v_ideal = np.zeros(M_crit)
    v_anti = np.zeros(M_crit)
    
    for j in range(M_crit):
        col = V[:, j]
        if j in max_obj:
            v_ideal[j] = col.max()
            v_anti[j] = col.min()
        else:
            v_ideal[j] = col.min()
            v_anti[j] = col.max()
            
    # 4. Cálculo de distancias euclidianas
    d_pos = np.linalg.norm(V - v_ideal, axis=1)
    d_neg = np.linalg.norm(V - v_anti, axis=1)
    
    # 5. Similitud relativa
    C = d_neg / (d_pos + d_neg + 1e-12)
    best_idx = int(np.argmax(C))
    
    return best_idx, C, v_ideal, v_anti

# Alias con firma exacta del notebook para retrocompatibilidad
def TOPSIS(F, w, max_obj, min_obj):
    best_idx, C, _, _ = topsis(F, w, max_obj, min_obj)
    return best_idx, C


def format_scientific(val: float, decimals: int = 1) -> str:
    """Formatea un número en notación científica clara (ej. 5.7 * 10^-5)."""
    if val == 0:
        return "0"
    s = f"{val:.{decimals}e}"
    m, e = s.split('e')
    return f"{float(m):.{decimals}f} · 10^{int(e)}"


def build_comparison_dataframe(rows_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Construye una tabla formateada de resultados MCDM para presentación / reporte.
    
    Cada elemento en rows_data es un dict con:
    - 'metodo': str (ej. 'NSGAII-TOPSIS', 'DFA')
    - 'pesos': str o tuple (ej. '0.8 - 0.2')
    - 'G_kg_h': float (Caudal de aire)
    - 'rd_m': float (Radio de gota)
    - 'energia_kw': float (Gasto energético)
    - 'humedad': float (Humedad final)
    """
    cols = [
        "Método", "Pesos",
        "Caudal de aire G [kg/h]", "Radio de gota r_d [m]",
        "Gasto energético (kW)", "Contenido de agua (kg/kg)"
    ]
    formatted_rows = []
    for r in rows_data:
        formatted_rows.append([
            r.get("metodo", ""),
            str(r.get("pesos", "")),
            f"{float(r.get('G_kg_h', 0)):.0f}",
            format_scientific(float(r.get("rd_m", 0))),
            f"{float(r.get('energia_kw', 0)):.3f}",
            f"{float(r.get('humedad', 0)):.3f}"
        ])
    return pd.DataFrame(formatted_rows, columns=cols)
