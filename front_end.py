import streamlit as st
import requests
import base64
from PIL import Image
import io

# Function to decode base64 to image
def decode_image(base64_string):
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    return img

# Streamlit page configuration
st.set_page_config(page_title='Auto Inpaint Anything Demo', layout='wide')

# Title for your app
st.title('Auto Inpaint Anything Demo')

# Sidebar for user inputs
with st.sidebar:
    st.header('Attach an image to be Segmented')
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    text_prompt = st.text_input("Choose Object To Detect:")
    submit_button = st.button('Detect and Segment wanted objects')
    st.header('Push on the button to remove Segmented Object')
    remove_object_button = st.button('Remove Detected Object')  # New button for removing objects
    st.header('Inform us about your new background')
    replace_prompt = st.text_input("How shall we change the Background?")
    replace_button = st.button('Replace Background')  # New button for removing objects
    st.header('Inform us to change your object')
    fill_prompt = st.text_input("How shall we change the object?")
    fill_button = st.button('Change the object as you want')  # New button for removing objects

# Define the API URLs
dino_sam_url = 'http://localhost:5004/app/demo/dino_sam'
remove_anything_url = 'http://localhost:5004/app/demo/remove_anything'
replace_anything_url = 'http://localhost:5004/app/demo/replace_anything'
fill_anything_url = 'http://localhost:5004/app/demo/fill_anything'

# Process actions based on button presses
if submit_button and uploaded_file and text_prompt:
    files = {'image': (uploaded_file.name, uploaded_file, uploaded_file.type)}
    data = {'text': text_prompt}
    response = requests.post(dino_sam_url, files=files, data=data)
    if response.status_code == 200:
        st.session_state.result = response.json()
        result_image = decode_image(st.session_state.result['segmented_result'])
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Submitted Image')
            st.image(uploaded_file, caption='Uploaded Image', width=800)
        with col2:
            st.subheader('Result Image')
            st.image(result_image, caption='Segmented Result Image', width=800)
    else:
        st.error('Failed to process the image')

if remove_object_button and uploaded_file and 'result' in st.session_state:
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    point_coords = st.session_state.result.get('point_coords', [])
    bbox_coords = st.session_state.result.get('bbox_coords', [])
    files = {'original_image': (uploaded_file.name, file_bytes, uploaded_file.type)}
    data = {'point_coords': str(point_coords), 'bbox_coords': bbox_coords}
    remove_response = requests.post(remove_anything_url, files=files, data=data)
    if remove_response.status_code == 200:
        remove_result = remove_response.json()
        removed_image_result = decode_image(remove_result['image_result'])
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Original Image')
            st.image(uploaded_file, caption='Uploaded Image', width=800)
        with col2:
            st.subheader('Removed Object Result')
            st.image(removed_image_result, caption='Removed Object Result', width=800)
    else:
        st.error('Failed to remove object from image')

if replace_button and uploaded_file and 'result' in st.session_state and replace_prompt:
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    point_coords = st.session_state.result.get('point_coords', [])
    bbox_coords = st.session_state.result.get('bbox_coords', [])
    files = {'original_image': (uploaded_file.name, file_bytes, uploaded_file.type)}
    data = {'point_coords': str(point_coords), 'text_prompt': replace_prompt, 'bbox_coords': bbox_coords}
    replace_response = requests.post(replace_anything_url, files=files, data=data)
    if replace_response.status_code == 200:
        replace_result = replace_response.json()
        replaced_image_result = decode_image(replace_result['image_result'])
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Original Image')
            st.image(uploaded_file, caption='Uploaded Image', width=800)
        with col2:
            st.subheader('Replaced Background Result')
            st.image(replaced_image_result, caption='Replaced Background Result', width=800)
    else:
        st.error('Failed to replace background')

if fill_button and uploaded_file and 'result' in st.session_state and fill_prompt:
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    point_coords = st.session_state.result.get('point_coords', [])
    bbox_coords = st.session_state.result.get('bbox_coords', [])
    files = {'original_image': (uploaded_file.name, file_bytes, uploaded_file.type)}
    data = {'point_coords': str(point_coords), 'text_prompt': fill_prompt, 'bbox_coords': bbox_coords}
    fill_response = requests.post(fill_anything_url, files=files, data=data)
    if fill_response.status_code == 200:
        fill_result = fill_response.json()
        filled_image_result = decode_image(fill_result['image_result'])
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Original Image')
            st.image(uploaded_file, caption='Uploaded Image', width=800)
        with col2:
            st.subheader('Changed Object Result')
            st.image(filled_image_result, caption='Changed Object Result', width=800)
    else:
        st.error('Failed to change object in image')
