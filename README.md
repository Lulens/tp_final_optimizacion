# Optimizador de grilla · Lollapalooza Argentina 2026

Trabajo de Aplicación Final · Optimización · UCA · 2026
Integrantes: Guadalupe Nogueira, Lucia Baez, Melanie Sampayo

**App online:** https://lolla-optimizador.streamlit.app

## Problema
Asignar los 72 artistas del festival a un día, escenario y horario, respetando las restricciones operativas (montaje, capacidad, headliners, horarios), para minimizar la superposición de artistas de géneros similares (ponderada por popularidad) y equilibrar el atractivo entre jornadas.

## Resultados principales
Frente a la grilla real anunciada:
- Conflicto entre artistas similares: −92,5 %
- Pares relevantes solapados: de 2 a 0
- Desequilibrio de atractivo entre días: de 136,7 a 0

## Archivos
| Archivo | Contenido |
|---|---|
| `app.py` | Prototipo en Streamlit |
| `lolla_opt.py` | Modelo CP-SAT (OR-Tools), evaluación y sensibilidad |
| `Lollapalooza_TAF.ipynb` | Formulación, resultados, sensibilidad y robustez |
| `lollapalooza_2026_dataset_completo_popularidad.xlsx` | Dataset (la app acepta otro con la misma estructura) |
| `requirements.txt` | Dependencias |

## Uso local
pip install -r requirements.txt
streamlit run app.py

## Estructura del Excel
Hojas obligatorias: `Artistas_72`, `Disponibilidad_dias`, `Escenarios`, `Dias_y_franjas`, `Parametros`. Opcional: `Pares_similitud`.
