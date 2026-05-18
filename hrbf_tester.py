import json
import streamlit as st
import numpy as np
import plotly.graph_objects as go

from rbf import RBF


st.set_page_config(layout="wide", page_title="HRBF Explorer")

# SESSION 

if "points" not in st.session_state:
    st.session_state.points = [
        {"px": 0.0, "py": 0.0, "alpha": 1.0, "bx": 0.0, "by": 0.0}
    ]

if "selected" not in st.session_state:
    st.session_state.selected = 0

if "rbf" not in st.session_state:
    st.session_state.rbf = "pow3"

if "param" not in st.session_state:
    st.session_state.param = None

if "last_upload_id" not in st.session_state:
    st.session_state.last_upload_id = None


points = st.session_state.points
rbf = RBF.get(st.session_state.rbf)
param = st.session_state.param

# hrbf
def f(X, Y, points):
    result = np.zeros_like(X, dtype=float)
    for p in points:
        DX = X - p["px"]
        DY = Y - p["py"]
        n  = np.maximum(np.sqrt(DX**2 + DY**2), 1e-8)
        result += p["alpha"] * rbf(n, param) + rbf.d(n, param) * (p["bx"] * DX / n + p["by"] * DY / n)
    return result


# LAYOUT

left, right = st.columns(2)

with left:
    grid = st.slider("Grid ", 1, 100, 10, step=1)
with right:
    res = st.slider("Resolution", 40, 300, 100, step=10)


left, right = st.columns([1, 2.5], gap="large")

with left:

    st.subheader("RBF")

    _rbf_names = RBF.names()
    _rbf_index = _rbf_names.index(st.session_state.rbf) if st.session_state.rbf in _rbf_names else 0

    selected_rbf = st.selectbox(
            "fonction",
            options=_rbf_names,
            index=_rbf_index,
            format_func=lambda n: n,
            )

    if selected_rbf != st.session_state.rbf:
        st.session_state.param = None

    st.session_state.rbf = selected_rbf
    rbf = RBF.get(selected_rbf)

    if rbf.extra_param is not None :
        _param_val = st.session_state.param
        if _param_val is None:
            _param_val = rbf.extra_param["default"]
        _param_val = max(rbf.extra_param["min"], min(rbf.extra_param["max"], _param_val))
        param = st.slider(
                rbf.extra_param["name"],
                min_value=rbf.extra_param["min"],
                max_value=rbf.extra_param["max"],
                value=_param_val,
                step=rbf.extra_param["step"]
                )
        st.session_state.param = param
    else:
        param = None
        st.session_state.param = None

    st.divider()
    

    st.subheader("Control Points")

    ic = st.session_state.last_upload_id # id pour empecher le cache streamlit d'override les points issu d'un upload

    # Point list as buttons
    for i, p in enumerate(points):
        label = f"{'>' if i == st.session_state.selected else ''}P{i+1}  ({p['px']:.2f}, {p['py']:.1f})"
        if st.button(label, key=f"sel_{i}_{ic}", width="stretch"):
            st.session_state.selected = i

    col_add, col_del = st.columns(2)
    with col_add:
        if st.button("+ point", width="stretch"):
            st.session_state.points.append({"px": 0.0, "py": 0.0, "alpha": 1.0, "bx": 0.0, "by": 0.0})
            st.session_state.selected = len(st.session_state.points) - 1
            st.rerun()
    with col_del:
        if st.button("Delete", width="stretch", disabled=len(points) <= 1):
            st.session_state.points.pop(st.session_state.selected)
            st.session_state.selected = max(0, st.session_state.selected - 1)
            st.rerun()

    st.divider()

    # Editor for selected point
    idx = st.session_state.selected
    p   = st.session_state.points[idx]

    st.markdown(f"**Point {idx + 1}**")

    c1, c2 = st.columns(2)
    with c1:
        p["px"] = st.number_input("x", value=p["px"], step=0.5, format="%.2f", key=f"px_{idx}_{ic}")
    with c2:
        p["py"] = st.number_input("y", value=p["py"], step=0.5, format="%.2f", key=f"py_{idx}_{ic}")

    p["alpha"] = st.slider("α",  min_value=-5.0, max_value=5.0, value=p["alpha"], step=0.01, key=f"a_{idx}_{ic}")
    p["bx"]    = st.slider("βx", min_value=-5.0, max_value=5.0, value=p["bx"],    step=0.01, key=f"bx_{idx}_{ic}")
    p["by"]    = st.slider("βy", min_value=-5.0, max_value=5.0, value=p["by"],    step=0.01, key=f"by_{idx}_{ic}")

    st.session_state.points[idx] = p


    # DETAIL DE F 

    st.divider()


    st.latex(r"""
        f(x) = \sum_{i}^N f_i
        \newline
        f_i = \alpha_i \phi(\| x - p_i \|) + \phi'(x) \times \beta_i \cdot \frac{ x - p_i }{\| x - p_i \|}
    """)


    cx = st.slider("x", float(-grid), float(grid), 0.0, step=0.01)
    cy = st.slider("y", float(-grid), float(grid), 0.0, step=0.01)


    rows = []
    total = 0

    for i, p in enumerate(st.session_state.points):
        dx = cx - p["px"] 
        dy = cy - p["py"]

        n = max(np.sqrt(dx**2 + dy**2), 1e-8) # x-pi

        phi_v = float(rbf(np.array([n]), param)[0])
        dphi_v = float(rbf.d(np.array([n]), param)[0])

        contrib = p["alpha"] * phi_v + dphi_v * (p["bx"] * dx / n + p["by"] * dy / n)
        total += contrib

        rows.append((i+1, p["px"], p["py"], n, p["alpha"], phi_v, dphi_v, p["bx"], dx/n, p["by"], dy/n, contrib))

    st.latex(rf"f(({cx:.2f}, {cy:.2f})) = {total:.2f}")

    for (i, px, py, n, alpha, phiv, dphiv, bx, dx_n, by, dy_n, contrib) in rows:
        latex = rf"""
        p_{i} = ({px}, {py}), \alpha_{i} = {alpha}, \beta_{i} = ({bx}, {by}) \newline 
        \begin{{aligned}}
        n_{i} &= \| x - p_{i} \| = {n:.2f} \\
                \phi(n_{i}) &= {phiv:.2f} \\
                \phi'(n_{i}) &= {dphiv:.2f} \\
        f_{i} &= {alpha:.2f} \times \phi(n_{i}) + \phi'(n_{i}) \times ({bx:.2f} \times {dx_n:.2f} + {by:.2f} \times {dy_n:.2f}) \\
        f_{i} &= {alpha:.2f} \times {phiv:.2f} + {dphiv:.2f} \times ({bx:.2f} \times {dx_n:.2f} + {by:.2f} \times {dy_n:.2f}) \\
        f_{i} &= {alpha * phiv:.2f} + {dphiv:.2f} \times ({bx * dx_n + by * dy_n:.2f}) \\
        f_{i} &= {alpha * phiv:.2f} + {dphiv * (bx * dx_n + by * dy_n):.2f} \\
        f_{i} &= {contrib:.2f} \\
        \end{{aligned}}
        """

        st.latex(latex)



# PLOT 

with right:
    x = np.linspace(-grid, grid, res)
    y = np.linspace(-grid, grid, res)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y, st.session_state.points)

    # Scatter markers for control points
    px_arr = [p["px"] for p in points]
    py_arr = [p["py"] for p in points]
    labels = [f"P{i+1} α={p['alpha']:.2f} βx={p['bx']:.2f} βy={p['by']:.2f}" for i, p in enumerate(points)]

    fig = go.Figure()
    fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale="RdBu", opacity=0.9, showscale=False))

    fig.add_trace(go.Surface(
        x=X, y=Y, z=np.zeros_like(Z),
        colorscale=[[0, "rgba(255,255,255,0.08)"], [1, "rgba(255,255,255,0.08)"]],
        showscale=False, hoverinfo="skip", name="z=0 plane",

        ))

    fig.add_trace(go.Scatter3d(
        x=px_arr, y=py_arr, z=np.zeros(len(points)),
        mode="markers+text",
        text=[f"P{i+1}" for i in range(len(points))],
        textposition="top center",
        hovertext=labels,
        hoverinfo="text",
        marker=dict(size=7, color="gold", line=dict(color="black", width=1)),
    ))


    # Highlight selected point
    sp = points[st.session_state.selected]
    fig.add_trace(go.Scatter3d(
        x=[sp["px"]], y=[sp["py"]], z=[0],
        mode="markers",
        marker=dict(size=11, color="red", line=dict(color="white", width=1)),
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter3d(
        x = [cx], y = [cy], z=[0],
        mode="markers",
        marker=dict(size=8, color="white", symbol="diamond", line=dict(color="white", width=2)),
        hoverinfo="skip"
        ))

    fig.update_layout(
        height=650,
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="f(x,y)"),
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False,
    )


    st.plotly_chart(fig, width="stretch")


    # contour 

     
    # Zt = np.sign(Z) * np.log1p(np.abs(Z))

    fig2 = go.Figure()
 
    fig2.add_trace(go.Contour(
        x=x, y=y, z=Z,
        contours=dict(coloring="lines", showlines=True),
        line=dict(width=1),
        showscale=True,
    ))
 
    fig2.add_trace(go.Contour(
        x=x, y=y, z=Z,
        contours=dict(start=0, end=0, size=1, coloring="none"),
        line=dict(color="red", width=2),
        showscale=False,
    ))
 
    fig2.add_trace(go.Scatter(
        x=px_arr, y=py_arr,
        mode="markers+text",
        text=[f"P{i+1}" for i in range(len(points))],
        textposition="top center",
        marker=dict(size=8, color="gold", line=dict(color="black", width=1)),
        showlegend=False,
        hovertext=labels,
        hoverinfo="text",
        ))




    fig2.add_trace(go.Scatter(
        x = [cx], y = [cy],
        mode="markers",
        marker=dict(size=8, color="white", symbol="cross-thin", line=dict(color="white", width=2)),
        showlegend=False,
        hoverinfo="skip"

        ))


 
    annotations = []
    for p in st.session_state.points:
        if p["bx"] != 0.0 or p["by"] != 0.0:
            annotations.append(dict(
                x=p["px"] + p["bx"],
                y=p["py"] + p["by"],
                ax=p["px"],
                ay=p["py"],
                xref="x", yref="y",
                axref="x", ayref="y",
                showarrow=True,
                arrowhead=1,
                arrowsize=1.0,
                arrowwidth=3.0,
                arrowcolor="yellow",
            ))
 
    fig2.update_layout(
        height=500,
        xaxis=dict(title="x", range=[-grid, grid]),
        yaxis=dict(title="y", range=[-grid, grid], scaleanchor="x"),
        annotations=annotations,
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor="rgb(17,17,34)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
 
    st.plotly_chart(fig2, width="stretch")



    # RBF PROFIL


    st.subheader("RBF Profil")
    r_max = grid * 1.5
    r_vals = np.linspace(0, r_max, 400)
    phi_vals = rbf(r_vals, param);
    dphi_vals = rbf.d(r_vals, param)




    
    col_phi, col_dphi = st.columns(2)
 
    with col_phi:
        fig_phi = go.Figure()
        fig_phi.add_trace(go.Scatter(
            x=r_vals, y=phi_vals,
            mode="lines", line=dict(color="royalblue", width=2), name="φ"
        ))
        fig_phi.add_hline(y=0, line=dict(color="white", width=1, dash="dot"))
        fig_phi.update_layout(
            title="φ(r)",
            height=300,
            margin=dict(l=0, r=0, t=40, b=0),
            xaxis_title="r",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgb(17,17,34)",
        )
        st.plotly_chart(fig_phi, width="stretch")
 
    with col_dphi:
        fig_dphi = go.Figure()
        fig_dphi.add_trace(go.Scatter(
            x=r_vals, y=dphi_vals,
            mode="lines", line=dict(color="tomato", width=2), name="φ'"
        ))
        fig_dphi.add_hline(y=0, line=dict(color="white", width=1, dash="dot"))
        fig_dphi.update_layout(
            title="φ'(r)",
            height=300,
            margin=dict(l=0, r=0, t=40, b=0),
            xaxis_title="r",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgb(17,17,34)",
        )
        st.plotly_chart(fig_dphi, width="stretch")

st.divider()
# export / export

left, right = st.columns(2)

with left:
    data = {
            "rbf": st.session_state.rbf,
            "param": st.session_state.param,
            "points": st.session_state.points
            }
    json_str = json.dumps(data, indent=2)
    st.download_button("Save", data=json_str, file_name="hrbf.json", mime="application/json")

with right:
    uploaded = st.file_uploader("Load", type="json")
    if uploaded is not None and uploaded.file_id != st.session_state.last_upload_id:
        st.session_state.last_upload_id = uploaded.file_id
        data = json.load(uploaded)
        st.session_state.rbf = data["rbf"]
        st.session_state.param = data.get("param")
        st.session_state.points = data["points"]
        st.session_state.selected = 0
        st.rerun()


