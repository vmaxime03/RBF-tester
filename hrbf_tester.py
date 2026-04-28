import streamlit as st
import numpy as np
import plotly.graph_objects as go

from rbf import RBF

st.set_page_config(layout="wide", page_title="HRBF Explorer")

# session state
if "points" not in st.session_state:
    st.session_state.points = [
        {"px": 0.0, "py": 0.0, "alpha": 1.0, "bx": 0.0, "by": 0.0}
    ]
if "selected" not in st.session_state:
    st.session_state.selected = 0

if "rbf" not in st.session_state:
    st.session_state.rbf = "pow3"

points = st.session_state.points

rbf = RBF.get(st.session_state.rbf)
param = None

# hrbf
def f(X, Y, points):
    result = np.zeros_like(X, dtype=float)
    for p in points:
        DX = X - p["px"]
        DY = Y - p["py"]
        n  = np.maximum(np.sqrt(DX**2 + DY**2), 1e-8)
        result += p["alpha"] * rbf(n, param) + rbf.d(n, param) * (p["bx"] * DX / n + p["by"] * DY / n)
    return result


# layout

left, right = st.columns([1, 2.5], gap="large")

with left:

    st.subheader("RBF")

    selected_rbf = st.selectbox(
            "fonction",
            options=RBF.names(),
            format_func = lambda n:n,
            )

    st.session_state.rbf = selected_rbf
    rbf = RBF.get(selected_rbf)

    if rbf.extra_param is not None :
        print(rbf.extra_param["min"])
        param = st.slider(
                rbf.extra_param["name"],
                min_value=rbf.extra_param["min"],
                max_value=rbf.extra_param["max"],
                value=rbf.extra_param["default"],
                step=rbf.extra_param["step"]
                )

    

    st.subheader("Control Points")

    # Point list as buttons
    for i, p in enumerate(points):
        label = f"{'>' if i == st.session_state.selected else ''}P{i+1}  ({p['px']:.1f}, {p['py']:.1f})"
        if st.button(label, key=f"sel_{i}", width="stretch"):
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
        p["px"] = st.number_input("x", value=p["px"], step=0.5, format="%.2f", key=f"px_{idx}")
    with c2:
        p["py"] = st.number_input("y", value=p["py"], step=0.5, format="%.2f", key=f"py_{idx}")

    p["alpha"] = st.slider("α",  min_value=-5.0, max_value=5.0, value=p["alpha"], step=0.05, key=f"a_{idx}")
    p["bx"]    = st.slider("βx", min_value=-5.0, max_value=5.0, value=p["bx"],    step=0.05, key=f"bx_{idx}")
    p["by"]    = st.slider("βy", min_value=-5.0, max_value=5.0, value=p["by"],    step=0.05, key=f"by_{idx}")

    st.session_state.points[idx] = p



# plot
with right:
    grid = st.slider("Grid ", 1, 100, 10, step=1)
    res = st.slider("Resolution", 40, 300, 100, step=10)
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
        xaxis=dict(title="x", range=[-10, 10]),
        yaxis=dict(title="y", range=[-10, 10], scaleanchor="x"),
        annotations=annotations,
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor="rgb(17,17,34)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
 
    st.plotly_chart(fig2, width="stretch")


