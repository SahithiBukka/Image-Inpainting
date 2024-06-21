

import streamlit as st
import pathlib
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import io
import base64
from PIL import Image

from random import randint
import numpy as np
import pickle
import streamlit as st
import base64
import streamlit as st

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:background/background.avif;base64,%s");
    background-position: center;
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown('<style>h1 { color: Black; }</style>', unsafe_allow_html=True)

    st.markdown('<style>p { color: Orange; }</style>', unsafe_allow_html=True)
    st.markdown(page_bg_img, unsafe_allow_html=True)
set_background('background/blbg.jpg')

STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static'

# We create a downloads directory within the streamlit static asset directory
# and we write output files to it.
DOWNLOADS_PATH = (STREAMLIT_STATIC_PATH / "downloads")
if not DOWNLOADS_PATH.is_dir():
    DOWNLOADS_PATH.mkdir()


def get_image_download_link(img, filename, text):
    """Generates a link to download a particular image file."""
    buffered = io.BytesIO()
    img.save(buffered, format='JPEG')
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    #st.write(href)
    return href


# Set title.
st.sidebar.title('Image Inpainting')


# Specify canvas parameters in application
uploaded_file = st.sidebar.file_uploader("Upload Image to restore:", type=["png", "jpg"])
image = None
res = None

if uploaded_file is not None:

    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 5)
    h, w = image.shape[:2]
    if w > 800:
        h_, w_ = int(h * 800 / w), 800
    else:
        h_, w_ = h, w

    canvas_result = st_canvas(
        fill_color='black',
        stroke_width=stroke_width,
        stroke_color='white',
        background_image=Image.open(uploaded_file).resize((h_, w_)),
        update_streamlit=True,
        height=h_,
        width=w_,
        drawing_mode='freedraw',
        key="canvas",
    )
    stroke = canvas_result.image_data

    if stroke is not None:
        #st.write(stroke.shape)
        if st.sidebar.checkbox('show mask'):
            st.image(stroke)

        mask = cv2.split(stroke)[3]
        mask = np.uint8(mask)
        mask = cv2.resize(mask, (w, h))

    st.sidebar.caption(' selection?')
    option = st.sidebar.selectbox('Mode', ['None',  'GAN'])

    

    if option == 'GAN':
        st.subheader('Result of GAN')
        res = cv2.inpaint(src=image, inpaintMask=mask, inpaintRadius=3, flags=cv2.INPAINT_NS)[:,:,::-1]
        st.image(res)
    else:
        pass

    if res is not None:
        # Display link.
        result = Image.fromarray(res)
        st.sidebar.markdown(
            get_image_download_link(result, 'output.png', 'Download Output'),
            unsafe_allow_html=True)
        #st.write(res.shape)
        #st.write(type(result))