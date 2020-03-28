import streamlit as st
import optical_flow as of
import time
import matplotlib.pyplot as plt

N = 16
threshold = 0.5

video = of.load_gif_to_array("seq.gif")
optical_f = of.kl_optical_flow_calc("seq.gif",N,threshold)
fig, ax, anim =  of.video_quiver(video,optical_f,N,threshold)

st.write("""
# Optical flow demo app
""")

st.pyplot(dpi=100)
while True:
    time.sleep(0.04)
    anim.new_frame_seq()
