from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
from io import BytesIO
from PIL import Image
import fitz  # PyMuPDF
import google.generativeai as genai
import dotenv
 
dotenv.load_dotenv()
 
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from the React frontend
 
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
 
def stream_llm_response(api_key=None):
    response_message = ""
 
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")
 
    gemini_messages = messages_to_gemini(st.session_state.get("messages", []))
 
    response_chunks = []
    for chunk in model.generate_content(
        contents=gemini_messages,
        stream=True,
    ):
        chunk_text = chunk.text or ""
        response_message += chunk_text
        response_chunks.append(chunk_text)
 
    st.session_state.setdefault("messages", []).append({
        "role": "assistant", 
        "content": [
            {
                "type": "text",
                "text": response_message,
            }
        ]})
 
    return response_chunks
 
def get_image_base64(image_raw):
    buffered = BytesIO()
    img_format = image_raw.format if image_raw.format else 'JPEG'
    try:
        image_raw.save(buffered, format=img_format)
    except ValueError:
        image_raw.convert('RGB').save(buffered, format='JPEG')
    img_byte = buffered.getvalue()
    return base64.b64encode(img_byte).decode('utf-8')
 
def base64_to_image(base64_string):
    base64_string = base64_string.split(",")[1]
    return Image.open(BytesIO(base64.b64decode(base64_string)))
 
def pdf_to_images(pdf_file):
    images = []
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images
 
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400
 
    file = request.files['image']
    img_bytes = file.read()
    raw_img = Image.open(BytesIO(img_bytes))
    img_type = file.mimetype
    img = get_image_base64(raw_img)
    message = {
        "role": "user",
        "content": [{
            "type": "image_url",
            "image_url": {"url": f"data:{img_type};base64,{img}"}
        }]
    }
    # Add message to session (simulate state management)
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    st.session_state["messages"].append(message)
 
    return jsonify({"status": "success"})
 
@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'pdf' not in request.files:
        return jsonify({"error": "No file part"}), 400
 
    file = request.files['pdf']
    images = pdf_to_images(file)
    messages = []
    for img in images:
        img_type = "image/jpeg"
        img_base64 = get_image_base64(img)
        messages.append({
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {"url": f"data:{img_type};base64,{img_base64}"}
            }]
        })
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    st.session_state["messages"].extend(messages)
 
    return jsonify({"status": "success"})
 
@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
 
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    st.session_state["messages"].append(
        {
            "role": "user", 
            "content": [{
                "type": "text",
                "text": prompt,
            }]
        }
    )
 
    response_chunks = stream_llm_response(api_key=os.getenv("GOOGLE_API_KEY"))
    return jsonify({"response": ''.join(response_chunks)})
 
if __name__ == "__main__":
    app.run(debug=True, port=5000)