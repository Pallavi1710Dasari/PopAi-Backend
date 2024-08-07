import streamlit as st
import dotenv
import os
from PIL import Image
import base64
from io import BytesIO
import google.generativeai as genai
import requests

dotenv.load_dotenv()

# Function to convert the messages format from OpenAI and Streamlit to Gemini
def messages_to_gemini(messages):
    gemini_messages = []
    prev_role = None
    for message in messages:
        if prev_role and (prev_role == message["role"]):
            gemini_message = gemini_messages[-1]
        else:
            gemini_message = {
                "role": "model" if message["role"] == "assistant" else "user",
                "parts": [],
            }

        for content in message["content"]:
            if content["type"] == "text":
                gemini_message["parts"].append(content["text"])
            elif content["type"] == "image_url":
                gemini_message["parts"].append(base64_to_image(content["image_url"]["url"]))

        if prev_role != message["role"]:
            gemini_messages.append(gemini_message)

        prev_role = message["role"]

    return gemini_messages

# Function to query and stream the response from the LLM
def stream_llm_response(api_key=None):
    response_message = ""

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")

    gemini_messages = messages_to_gemini(st.session_state.messages)

    for chunk in model.generate_content(
        contents=gemini_messages,
        stream=True,
    ):
        chunk_text = chunk.text or ""
        response_message += chunk_text
        yield chunk_text

    st.session_state.messages.append({
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": response_message,
            }
        ]})

# Function to convert file to base64 encoded string
def get_image_base64(image_raw):
    buffered = BytesIO()
    image_raw.save(buffered, format=image_raw.format)
    img_byte = buffered.getvalue()
    return base64.b64encode(img_byte).decode('utf-8')

def base64_to_image(base64_string):
    base64_string = base64_string.split(",")[1]
    return Image.open(BytesIO(base64.b64decode(base64_string)))

def url_to_base64(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    img_type = img.format.lower()
    return get_image_base64(img), f"image/{img_type}"

def main():

    # --- Page Config ---
    st.set_page_config(
        page_title="Math Solver",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # --- Header ---
    st.html("""<h1 style="text-align: center; color: #6ca395;"> <i>The Maths Solver</i> </h1>""")

    # --- Side Bar ---
    with st.sidebar:
        # Menu text at the top
        st.markdown("## 📋 Menu")
        st.write("Welcome to the Maths Solver app! Use the options below to interact with the app.")
       
    # --- Main Content ---
    # Retrieve API key from .env file
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not google_api_key:
        st.write("#")
        st.warning("⬅️ Google API Key is missing from .env file. Please check your .env file.")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Displaying the previous messages if there are any
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                for content in message["content"]:
                    if content["type"] == "text":
                        st.write(content["text"])
                    elif content["type"] == "image_url":      
                        st.image(content["image_url"]["url"])

        # Side bar model options and inputs
        with st.sidebar:

            st.divider()

            def reset_conversation():
                if "messages" in st.session_state and len(st.session_state.messages) > 0:
                    st.session_state.pop("messages", None)

            st.button(
                "🗑️ Reset conversation",
                on_click=reset_conversation,
            )

            st.divider()

            # Image Upload
            st.write(f"### **🖼️ Add an image:**")

            def add_image_to_messages():
                if st.session_state.uploaded_img:
                    # Read image from file uploader
                    img_file = st.session_state.uploaded_img
                    img_bytes = img_file.read()
                    raw_img = Image.open(BytesIO(img_bytes))
                    img_type = img_file.type
                    img = get_image_base64(raw_img)
                    st.session_state.messages.append(
                        {
                            "role": "user",
                            "content": [{
                                "type": "image_url",
                                "image_url": {"url": f"data:{img_type};base64,{img}"}
                            }]
                        }
                    )
                elif "camera_img" in st.session_state and st.session_state.camera_img:
                    # Read image from camera input
                    img = st.session_state.camera_img
                    raw_img = Image.open(BytesIO(img))
                    img_type = "image/jpeg"  # Camera input is usually JPEG
                    img = get_image_base64(raw_img)
                    st.session_state.messages.append(
                        {
                            "role": "user",
                            "content": [{
                                "type": "image_url",
                                "image_url": {"url": f"data:{img_type};base64,{img}"}
                            }]
                        }
                    )
                elif st.session_state.url_img:
                    # Read image from URL
                    img_url = st.session_state.url_img
                    img, img_type = url_to_base64(img_url)
                    st.session_state.messages.append(
                        {
                            "role": "user",
                            "content": [{
                                "type": "image_url",
                                "image_url": {"url": f"data:{img_type};base64,{img}"}
                            }]
                        }
                    )

            cols_img = st.columns(2)

            with cols_img[0]:
                with st.popover("📁 Upload"):
                    st.file_uploader(
                        "Upload an image:",
                        type=["png", "jpg", "jpeg"],
                        accept_multiple_files=False,
                        key="uploaded_img",
                        on_change=add_image_to_messages,
                    )

            with cols_img[1]:                    
                with st.popover("📸 Camera"):
                    activate_camera = st.checkbox("Activate camera")
                    if activate_camera:
                        st.camera_input(
                            "Take a picture",
                            key="camera_img",
                            on_change=add_image_to_messages,
                        )

            st.divider()
            
            st.write(f"### **🌐 Add an image URL:**")

            st.text_input(
                "Image URL:",
                key="url_img",
                on_change=add_image_to_messages,
            )
           
        # Chat input
        if prompt := st.chat_input("Hi! Ask me anything..."):
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": prompt,
                    }]
                }
            )
           
            # Display the new messages
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                st.write_stream(
                    stream_llm_response(
                        api_key=google_api_key
                    )
                )

if __name__ == "__main__":
    main()
