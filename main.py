import streamlit as st
from edge_detection import EdgeDetect
from PIL import Image
import numpy as np


st.set_page_config("Edge Detection in CV", 
                   layout='wide', 
                   initial_sidebar_state='expanded')

st.title('Edge Detection in CV')
st.markdown('Explore Gradient and Gaussian edge detection methods.')


file = st.file_uploader('Upload image', type=['jpg', 'png'])
with st.sidebar:
    st.subheader('Edge Detection Methods:')
    option = st.radio('Select method: ',
             ['None', 'Sobel', 'Prewitt', 'Robert',
               'Laplacian of Gaussian', 'Canny detector',
               'Holistically-nested'])

# First convert image to array format. Based on method adjust parameters
# and get output of edge detection for different methods.
if file is not None:
    image = Image.open(file)
    image = np.asarray(image)
    method = EdgeDetect(image)
    if option == 'None':
        st.subheader('Select Edge Detection Method')
    elif option == 'Sobel':
        st.subheader('Sobel Edge Detection')
        t_val = st.slider('Select threshold: ', 0, 255, 20, 1)
        c1, c2 = st.columns(2)
        kernel = int(c1.radio('Select Kernel size:', ['3', '5']))
        opencv = c2.radio('Select Opencv:', [True, False])
        out1, out2, out3 = method.sobel_edge(thresh=t_val, 
                                             kernel_size=kernel, 
                                             opencv=opencv)
        col1, col2, col3 = st.columns(3)
        col1.image(out1, 'Sobel output', clamp=True)
        col2.image(out2, 'X gradient', clamp=True)
        col3.image(out3, 'Y gradient', clamp=True)
    elif option == 'Prewitt':
        st.subheader('Prewitt Edge Detection')
        t_val = st.slider('Select threshold: ', 0, 255, 20, 1)
        out1, out2, out3 = method.prewitt_edge(thresh=t_val)
        col1, col2, col3 = st.columns(3)
        col1.image(out1, 'Prewitt output', clamp=True)
        col2.image(out2, 'X gradient', clamp=True)
        col3.image(out3, 'Y gradient', clamp=True)
    elif option == 'Robert':
        st.subheader('Robert Edge Detection')
        t_val = st.slider('Select threshold: ', 0, 255, 20, 1)
        out1, out2, out3 = method.robert_edge(thresh=t_val)
        col1, col2, col3 = st.columns(3)
        col1.image(out1, 'Robert output', clamp=True)
        col2.image(out2, 'X gradient', clamp=True)
        col3.image(out3, 'Y gradient', clamp=True)
    elif option == 'Laplacian of Gaussian':
        st.subheader('Laplacian of Gaussian')
        kernel = int(st.radio('Select Kernel:', ['3', '5']))
        out = method.laplacian_edge(kernel)
        st.image(out, 'Laplacian of Gaussian', clamp=True)
    elif option == 'Canny detector':
        st.subheader('Canny Edge Detection')
        min_v = st.slider('Set minimum threshold:', 0, 255, 10, 1)
        max_v = st.slider('Set maximum threshold:', 0, 255, 50, 1)
        l2norm = st.radio('Select L2 gradient:', [True, False])
        c1, c2, c3 = st.columns(3)
        out1 = method.canny_edge(min_v, max_v, 3, l2norm)
        c1.image(out1, 'aperture size 3', clamp=True)
        out2 = method.canny_edge(min_v, max_v, 5, l2norm)
        c2.image(out2, 'aperture size 5', clamp=True)
        out3 = method.canny_edge(min_v, max_v, 7, l2norm)
        c3.image(out3, 'aperture size 7', clamp=True)
    else: 
        st.subheader('Holistically-hested Edge Detection')
        factor = st.slider('Select scale factor:', 0.0, 1.0, 0.5, 0.1)
        swap = st.radio('Select swapRB:', [True, False])
        c1, c2 = st.columns(2)
        out1, out2 = method.holistically_nested_edge(factor, swap)
        c1.image(out1, 'Blob output', clamp=True)
        c2.image(out2, 'HED output', clamp=True)
    
