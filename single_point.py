import math
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

import rbf as rbf

st.set_page_config(layout="wide")


# SLIDERS 

a = st.slider("alpha", -5.0, 5.0, 1.0)
bx = st.slider("beta.x", -5.0, 5.0, 1.0)
by = st.slider("beta.y", -5.0, 5.0, 1.0)



# DATA

x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)

def f(X, Y, a, bx, by):
    n = np.sqrt(X**2 + Y**2)
    return a * rbf.pow3(n) + rbf.dpow3(n) * (bx * X/n + by * Y/n)


Z = f(X, Y, a, bx, by)

minz = np.min(Z)
maxz = np.max(Z)


# pour que les contours soit proches les un des autres
Zt = np.sign(Z) * np.log1p(np.abs(Z))

fig = go.Figure()

fig.add_trace(
    go.Contour(
        x=x,
        y=y,
        z=Zt,
        contours=dict(
            coloring="lines",
            showlines=True
        ),
        line=dict(width=1),
        showscale=True
    )
)

fig.add_trace(
    go.Contour(
        x=x,
        y=y,
        z=Zt,
       contours=dict(
            start=0,
            end=0,
            size=1,
            coloring="none"
        ),
        line=dict(color="red", width=2),
        showscale=False
        )
)

# vecteur beta
fig.add_annotation(
        x=bx,
        y=by,
        ax=0.0,
        ay=0.0,
         xref="x",
    yref="y",
    axref="x",
    ayref="y",
    showarrow=True,
    arrowhead=1,
    arrowsize=1.0,
    arrowwidth=3.0,
    arrowcolor="yellow"
)

fig.update_layout(
    xaxis_title="x",
    yaxis_title="y",
    yaxis=dict(scaleanchor="x", scaleratio=1)
)



# SURF PLOT

fig2 = go.Figure(
    data=go.Surface(x=X, y=Y, z=Z)
)
fig2.update_layout(
    scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="f(x,y)")
)




# layout

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig, width="stretch")

with col2:
    st.plotly_chart(fig2, width="stretch")



