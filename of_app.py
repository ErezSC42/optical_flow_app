import streamlit as st
import optical_flow as of
import time
import matplotlib.pyplot as plt

#N = 16
st.sidebar.title("Algorithm Parameters")
threshold = st.sidebar.slider("threshold", min_value=0., max_value=1., step=0.01, key=0.1)
n_list = [2,4,8,16,32,64]
selected_n = st.sidebar.radio("N", [str(i) for i in n_list])
N = int(selected_n)

video = of.load_gif_to_array("seq.gif")
optical_f = of.kl_optical_flow_calc("seq.gif",N,threshold)
fig, ax, anim =  of.video_quiver(video,optical_f,N,threshold)

st.write("""
# Optical flow demo app
""")

st.pyplot(dpi=100)
