"""
Optimizador de grilla · Lollapalooza Argentina 2026  (TAF Optimización)
Ejecutar:  streamlit run app.py
"""
import io
import os
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import lolla_opt as lo

st.set_page_config(page_title="Optimizador Lollapalooza 2026", page_icon="🎪", layout="wide")
DEFAULT_FILE = "lollapalooza_2026_dataset_completo_popularidad.xlsx"


def sdf(df, **kw):
    """st.dataframe a ancho completo, compatible con versiones nuevas y viejas de Streamlit."""
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


def sed(df, **kw):
    try:
        return st.data_editor(df, width="stretch", **kw)
    except TypeError:
        return st.data_editor(df, use_container_width=True, **kw)


# ------------------------------------------------------------------ datos
@st.cache_data(show_spinner=False)
def get_data(raw: bytes):
    return lo.load_data(io.BytesIO(raw))


st.title("🎪 Optimizador de grilla · Lollapalooza Argentina 2026")
st.caption("Asigna cada artista a un día, escenario y horario minimizando la superposición de artistas "
           "similares (ponderada por popularidad) y equilibrando el atractivo entre jornadas. Modelo CP-SAT (OR-Tools).")

with st.sidebar:
    st.header("1 · Datos")
    up = st.file_uploader("Excel con el dataset (.xlsx)", type=["xlsx"])
    if up is not None:
        raw = up.getvalue()
        src = up.name
    elif os.path.exists(DEFAULT_FILE):
        raw = open(DEFAULT_FILE, "rb").read()
        src = f"{DEFAULT_FILE} (incluido)"
    else:
        st.info("Subí el Excel para comenzar.")
        st.stop()
    try:
        data0 = get_data(raw)
    except lo.DataError as e:
        st.error(str(e))
        st.stop()
    st.success(f"Cargado: {src}")

    st.header("2 · Parámetros")
    modo = st.radio("Modo", ["Flexible (optimiza día, escenario y hora)", "Fijo (conserva el día real)"])
    w_conf = st.slider("Peso de conflictos", 0.0, 3.0, float(data0.params.get("Peso_conflictos", 1.0)), 0.1)
    w_eq = st.slider("Peso de equilibrio entre días", 0.0, 3.0, float(data0.params.get("Peso_equilibrio", 0.2)), 0.05)
    transform = st.selectbox("Transformación de popularidad", ["lineal", "raiz", "log"],
                             help="Con 'lineal' unos pocos artistas concentran casi todo el costo.")
    with st.expander("Similitud entre géneros"):
        s_same = st.slider("Mismo género", 0.0, 1.0, float(data0.params.get("Similitud_mismo_genero", 1.0)), 0.05)
        s_near = st.slider("Géneros cercanos", 0.0, 1.0, float(data0.params.get("Similitud_generos_cercanos", 0.5)), 0.05)
        s_far = st.slider("Géneros distintos", 0.0, 1.0, float(data0.params.get("Similitud_generos_distintos", 0.1)), 0.05)
    with st.expander("Restricciones operativas"):
        use_cap = st.checkbox("Respetar capacidad de escenarios", True)
        cap_scale = st.slider("Capacidad de escenarios (%)", 70, 130, 100, 5) / 100
        hl_hour = st.slider("Headliners no antes de (hora)", 15.0, 22.0, 19.0, 0.5)
        max_hl = st.number_input("Máx. headliners por día (0 = sin límite)", 0, 10, 0)
        prep_extra = st.slider("Montaje extra entre shows (min)", 0, 30, 0, 5)
        art_delta = st.slider("Relajar mín/máx de artistas por día (±)", 0, 4, 0)
    st.header("3 · Resolución")
    tl = st.slider("Tiempo límite (seg)", 20, 300, 90, 10)

cfg = lo.Config(
    w_conf=w_conf, w_equil=w_eq, pop_transform=transform, sim_same=s_same, sim_near=s_near, sim_far=s_far,
    flexible_days=modo.startswith("Flexible"), use_capacity=use_cap, cap_scale=cap_scale,
    headliner_min_start=int(hl_hour * 60), max_headliners_day=(int(max_hl) or None),
    prep_extra=prep_extra, art_delta=art_delta,
)

# disponibilidad editable (what-if del usuario)
if st.session_state.get("raw_src") != (src, len(raw)):
    st.session_state["raw_src"] = (src, len(raw))
    st.session_state["avail"] = data0.avail.copy()
    st.session_state.pop("res", None)
data = replace(data0, avail=st.session_state["avail"].copy())

tab_d, tab_o, tab_c, tab_s, tab_m = st.tabs(["📋 Datos", "🚀 Optimizar", "⚖️ Real vs óptima", "🔍 Sensibilidad", "ℹ️ Modelo"])

# ------------------------------------------------------------------ datos
with tab_d:
    st.subheader("Validación del dataset")
    msgs = lo.validate(data, cfg)
    if not msgs:
        st.success("Sin observaciones.")
    for lvl, msg in msgs:
        {"error": st.error, "warn": st.warning, "info": st.info}[lvl](msg)
    for n in data.notes:
        st.caption("• " + n)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Artistas**")
        sdf(data.artists.drop(columns=["demand"]).rename(columns={
            "id": "ID", "name": "Artista", "genre": "Género", "dur": "Min", "hl": "Headliner", "pop": "Popularidad",
            "day0": "Día real (idx)", "stage0": "Escenario real", "start0": "Inicio real (min)"}), height=300)
    with c2:
        st.markdown("**Escenarios**")
        sdf(data.stages)
        st.markdown("**Jornadas**")
        sdf(data.days)
    st.markdown("**Disponibilidad por día** · editá y volvé a optimizar (what-if: ¿qué pasa si un artista no puede un día?)")
    dv = pd.DataFrame(st.session_state["avail"].astype(bool), columns=list(data.days["label"]))
    dv.insert(0, "Artista", data.artists["name"])
    edited = sed(dv, hide_index=True, height=300, disabled=["Artista"])
    new_av = edited[list(data.days["label"])].values.astype(int)
    if (new_av != st.session_state["avail"]).any():
        st.session_state["avail"] = new_av
        st.rerun()

# ------------------------------------------------------------------ optimizar
with tab_o:
    errs = [m for l, m in msgs if l == "error"]
    if errs:
        st.error("Corregí los errores de datos antes de optimizar.")
    elif st.button("🚀 Optimizar grilla", type="primary"):
        bar = st.progress(0, text="Resolviendo…")
        with st.spinner(f"CP-SAT trabajando (hasta {tl}s)…"):
            res = lo.solve_best(data, cfg, time_limit=tl)
        bar.progress(100, text="Listo")
        st.session_state["res"] = (res, cfg, data)

    if "res" not in st.session_state:
        st.info("Configurá los parámetros a la izquierda y presioná **Optimizar grilla**.")
    else:
        res, rcfg, rdata = st.session_state["res"]
        if not res.ok:
            st.error(f"{res.status}: {res.message}")
        else:
            b, m = res.baseline, res.metrics
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Reducción del conflicto", f"{100 * res.reduccion_conflicto:.1f} %", help="vs. grilla real, mismos parámetros")
            k2.metric("Pares relevantes solapados", m["pares_solapados_relevantes"],
                      f"{m['pares_solapados_relevantes'] - b['pares_solapados_relevantes']} vs real", delta_color="inverse")
            k3.metric("Desequilibrio entre días", f"{m['desequilibrio']:.0f}",
                      f"{m['desequilibrio'] - b['desequilibrio']:.0f} vs real", delta_color="inverse")
            k4.metric("Estado / tiempo", res.status, f"{res.seconds:.0f} s", delta_color="off")
            viol = lo.check_schedule(res.schedule, rdata, rcfg)
            if viol:
                st.error("La grilla viola restricciones:\n\n" + "\n".join("- " + v for v in viol))
            else:
                st.success("✔ Verificación independiente: la grilla cumple todas las restricciones.")
            with st.expander("Detalle de la resolución por fases"):
                sdf(pd.DataFrame(res.phases), hide_index=True)
                st.caption("El gap solo es significativo cuando el estado es OPTIMAL; en F2/F3 la cota inferior de un "
                           "problema de scheduling con linealizaciones es floja, por eso se valida contra la grilla real.")
            day = st.radio("Jornada", list(rdata.days["label"]), horizontal=True)
            di = list(rdata.days["label"]).index(day)
            fig = lo.gantt_figure(res.schedule, rdata, di, "· grilla optimizada")
            st.pyplot(fig)
            plt.close(fig)
            tbl = lo.full_schedule(res.schedule, rdata)
            sdf(tbl[tbl["day"] == di].drop(columns=["start", "day", "Día"]), hide_index=True)
            d1, d2 = st.columns(2)
            d1.download_button("⬇️ Descargar Excel", lo.to_excel_bytes(res.schedule, rdata, rcfg, res),
                               "grilla_optimizada_lollapalooza2026.xlsx")
            d2.download_button("⬇️ Descargar CSV", tbl.drop(columns=["start", "day"]).to_csv(index=False).encode("utf-8"),
                               "grilla_optimizada.csv")

# ------------------------------------------------------------------ comparación
with tab_c:
    if "res" not in st.session_state or not st.session_state["res"][0].ok:
        st.info("Primero generá una grilla optimizada.")
    else:
        res, rcfg, rdata = st.session_state["res"]
        base = lo.baseline_schedule(rdata)
        st.markdown("**Atractivo acumulado por jornada** (suma de popularidad transformada)")
        att = pd.DataFrame({"Real": res.baseline["atractivo_por_dia"], "Optimizada": res.metrics["atractivo_por_dia"]},
                           index=list(rdata.days["label"]))
        st.bar_chart(att)
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**Mayores superposiciones · grilla REAL**")
            sdf(lo.top_conflicts(base, rdata, rcfg, 8), hide_index=True)
        with cc2:
            st.markdown("**Mayores superposiciones · grilla OPTIMIZADA**")
            sdf(lo.top_conflicts(res.schedule, rdata, rcfg, 8), hide_index=True)
        day = st.radio("Ver grilla real de", list(rdata.days["label"]), horizontal=True, key="dreal")
        fig = lo.gantt_figure(base, rdata, list(rdata.days["label"]).index(day), "· grilla real")
        st.pyplot(fig)
        plt.close(fig)

# ------------------------------------------------------------------ sensibilidad
with tab_s:
    st.subheader("Análisis de sensibilidad y robustez")
    st.caption("En un modelo entero no hay precios sombra como en LP: se re-optimiza bajo cada variación y se compara "
               "contra la grilla real evaluada con los mismos parámetros.")
    catalogo = {
        "Base (parámetros actuales)": {},
        "Sin término de equilibrio (peso 0)": {"w_equil": 0.0},
        "Equilibrio ×5": {"w_equil": cfg.w_equil * 5},
        "Montaje +15 min": {"prep_extra": 15},
        "Capacidad −10 %": {"cap_scale": cfg.cap_scale * 0.9},
        "Capacidad −15 % (umbral de factibilidad)": {"cap_scale": cfg.cap_scale * 0.85},
        "Sin restricción de capacidad": {"use_capacity": False},
        "Similitud entre géneros cercanos 0.8": {"sim_near": 0.8},
        "Similitud entre géneros distintos 0.3": {"sim_far": 0.3},
        "Popularidad raíz cuadrada": {"pop_transform": "raiz"},
        "Popularidad logarítmica": {"pop_transform": "log"},
        "Más flexibilidad de artistas por día (±2)": {"art_delta": 2},
        "Modo fijo (días reales)": {"flexible_days": False},
    }
    sel = st.multiselect("Escenarios a evaluar", list(catalogo), default=list(catalogo)[:4])
    t_sc = st.slider("Tiempo por escenario (seg)", 15, 120, 40, 5)
    # arranque en caliente: si ya hay una grilla optimizada, todos los escenarios parten de ella.
    # Si además se optimizó con los MISMOS parámetros y datos, la fila Base la reutiliza tal cual.
    base_res, warm = None, None
    if "res" in st.session_state and st.session_state["res"][0].ok:
        r0, c0, d0 = st.session_state["res"]
        warm = r0.schedule
        if c0 == cfg and np.array_equal(d0.avail, data.avail):
            base_res = r0
        st.caption("✔ Los escenarios arrancan desde la grilla optimizada y nunca devuelven algo peor que ella "
                   "cuando sigue siendo factible" + (" · la fila Base reutiliza la corrida principal." if base_res else "."))
    else:
        st.caption("Tip: optimizá primero en la pestaña 🚀 para que los escenarios arranquen desde esa grilla "
                   "(resultados más estables y comparables).")
    if st.button("▶️ Ejecutar sensibilidad") and sel:
        prog = st.progress(0.0)
        rows = []
        for k, lab in enumerate(sel):
            prog.progress(k / len(sel), text=f"{lab}…")
            rows.append(lo.scenario_table(data, cfg, {lab: catalogo[lab]}, time_limit=t_sc,
                                          base_result=base_res, hint=warm))
        prog.progress(1.0, text="Listo")
        st.session_state["sens"] = pd.concat(rows, ignore_index=True)
    if "sens" in st.session_state:
        sdf(st.session_state["sens"], hide_index=True)
        st.download_button("⬇️ Descargar tabla", st.session_state["sens"].to_csv(index=False).encode("utf-8"), "sensibilidad.csv")

    st.markdown("---")
    st.markdown("**Frontera conflicto vs. equilibrio (ε-restricción):** minimiza el conflicto con el desequilibrio "
                "entre días acotado por ε. Se recorre ε de menor a mayor y cada punto parte del anterior, "
                "así la curva es monótona.")
    eps_txt = st.text_input("Valores de ε (separados por coma)", "5, 15, 30, 60, 100, 140")
    t_eps = st.slider("Tiempo por punto (seg)", 15, 120, 40, 5, key="t_eps")
    if st.button("📈 Calcular frontera"):
        try:
            eps = [float(e) for e in eps_txt.replace(";", ",").split(",") if e.strip()]
        except ValueError:
            st.error("Escribí números separados por coma.")
            eps = []
        if eps:
            with st.spinner(f"Resolviendo {len(eps)} puntos…"):
                st.session_state["front"] = lo.epsilon_frontier(data, cfg, eps, time_limit=t_eps, hint=warm)
    if "front" in st.session_state:
        fr = st.session_state["front"]
        sdf(fr, hide_index=True)
        ok = fr.dropna(subset=["Reducción conflicto vs real (%)"])
        if len(ok):
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(ok["Desequilibrio entre días"], ok["Reducción conflicto vs real (%)"], "o-")
            for _, r in ok.iterrows():
                ax.annotate(f"ε={r['ε (tope de desequilibrio)']:g}", (r["Desequilibrio entre días"],
                            r["Reducción conflicto vs real (%)"]), fontsize=7, xytext=(4, -10), textcoords="offset points")
            ax.set_xlabel("Desequilibrio entre días"); ax.set_ylabel("Reducción del conflicto vs real (%)")
            ax.grid(alpha=.3)
            st.pyplot(fig); plt.close(fig)
    st.markdown("---")
    st.markdown("**Robustez frente a error en la similitud** (evaluación fuera de muestra: la grilla no se re-optimiza)")
    sigma = st.slider("Desvío del ruido lognormal sobre cada costo de par", 0.1, 1.0, 0.3, 0.05)
    if "res" in st.session_state and st.session_state["res"][0].ok and st.button("🎲 Simular 300 escenarios"):
        res, rcfg, rdata = st.session_state["res"]
        rb = lo.robustness(rdata, rcfg, res.schedule, n=300, sigma=sigma)
        r1, r2, r3 = st.columns(3)
        r1.metric("Reducción media", f"{rb['reduccion_%'].mean():.1f} %")
        r2.metric("Peor 5 % de casos", f"{rb['reduccion_%'].quantile(0.05):.1f} %")
        r3.metric("Casos mejores que la real", f"{100 * (rb['reduccion_%'] > 0).mean():.0f} %")
        hist, edges = np.histogram(rb["reduccion_%"], bins=20)
        st.bar_chart(pd.DataFrame({"casos": hist}, index=[f"{e:.0f}" for e in edges[:-1]]))
    elif "res" not in st.session_state:
        st.caption("Generá una grilla optimizada para habilitar la simulación.")

# ------------------------------------------------------------------ modelo
with tab_m:
    st.markdown("""
**Decisiones.** Para cada artista *i*: día *d*, escenario *s* y minuto de inicio. Variables binarias `x[i,d,s]` (asignación) y entera `start[i]` (inicio, en slots).

**Restricciones.** (1) cada artista actúa una vez, en un día disponible · (2) ventana horaria de cada escenario y jornada ·
(3) sin superposición dentro de un escenario, incluyendo el montaje · (4) mínimo/máximo de artistas por día ·
(5) mínimo de headliners por día y horario mínimo de inicio · (6) demanda estimada ≤ capacidad del escenario.

**Objetivo.** Minimizar  `w_conf · Σ c_ij · horas_solapadas(i,j) / conflicto_real  +  w_eq · (máx_d A_d − mín_d A_d) / desequilibrio_real`,
con `c_ij = similitud(género_i, género_j) · f(pop_i) · f(pop_j)`. Ambos términos están normalizados por la grilla real, así que
**1.0 equivale a la grilla anunciada**.

**Resolución.** CP-SAT con intervalos opcionales + `NoOverlap` por escenario y día. En modo flexible se aplica una matheurística de 3 fases
(días reales → días libres con los pares más costosos → pulido con todos los pares).

**Limitaciones.** La similitud musical es un proxy (no audiencias observadas); Trends no está cargado; la disponibilidad es un supuesto;
la demanda se deriva de la popularidad; el modelo no contempla tiempos de traslado entre escenarios.
""")
