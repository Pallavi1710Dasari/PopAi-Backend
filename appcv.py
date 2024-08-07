from flask import Flask, request, jsonify
from flask_cors import CORS
import dotenv
import os
from PIL import Image
import base64
from io import BytesIO
import google.generativeai as genai
import fitz  # PyMuPDF

dotenv.load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

google_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Google Generative AI
genai.configure(api_key=google_api_key)
model = genai.GenerativeModel(model_name="gemini-1.5-pro")

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

@app.route('/generate', methods=['POST'])
def generate_response():
    data = request.json
    messages = data.get('messages', [])

    gemini_messages = messages_to_gemini(messages)
    response_message = ""
    for chunk in model.generate_content(contents=gemini_messages, stream=False):
        chunk_text = chunk.text or ""
        response_message += chunk_text

    return jsonify({"response": response_message})

@app.route('/upload_image', methods=['POST'])
def upload_image():
    file = request.files['file']
    img = Image.open(file.stream)
    img_base64 = get_image_base64(img)
    return jsonify({"image_base64": img_base64})

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    file = request.files['file']
    images = pdf_to_images(file)
    images_base64 = [get_image_base64(img) for img in images]
    return jsonify({"images_base64": images_base64})

if __name__ == '__main__':
    app.run(debug=True)
