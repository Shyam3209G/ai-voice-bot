# =========================================
# FASTAPI VOICE BOT - FINAL CODE
# =========================================

from fastapi import FastAPI, UploadFile, File
import whisper
from gtts import gTTS
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ----------- INIT APP -----------
app = FastAPI()

# ----------- LOAD ASR MODEL -----------
asr_model = whisper.load_model("base")

# ----------- DATASET -----------

texts = [
    "where is my order","track my order","order not delivered","where is shipment","check my order status",
    "cancel my order","i want to cancel","please cancel my order",
    "i want refund","refund my money","give my money back",
    "payment failed","payment not working","unable to pay",
    "late delivery","delivery is late","why is my order late",
    "cannot login","login problem","unable to login",
    "product details","need product info","tell me about product",
    "i have complaint","bad service","not happy with service",
    "hello","hi","good morning"
]

labels = [
    "order_status","order_status","order_status","order_status","order_status",
    "cancel_order","cancel_order","cancel_order",
    "refund_request","refund_request","refund_request",
    "payment_issue","payment_issue","payment_issue",
    "delivery_delay","delivery_delay","delivery_delay",
    "login_issue","login_issue","login_issue",
    "product_query","product_query","product_query",
    "complaint","complaint","complaint",
    "greeting","greeting","greeting"
]

# ----------- TRAIN MODEL -----------

vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)

# ----------- RESPONSES -----------

responses = {
    "order_status": "Your order is on the way.",
    "cancel_order": "Your order has been cancelled.",
    "refund_request": "Your refund will be processed soon.",
    "payment_issue": "Please check your payment details.",
    "delivery_delay": "We apologize for the delay.",
    "login_issue": "Please reset your password.",
    "product_query": "Please check product description.",
    "complaint": "We have registered your complaint.",
    "greeting": "Hello! How can I help you?"
}

# ----------- FUNCTIONS -----------

def predict_intent(text):
    x = vectorizer.transform([text])
    probs = model.predict_proba(x)
    intent = model.predict(x)[0]
    confidence = probs.max()
    return intent, confidence

def generate_response(intent, confidence):
    if confidence < 0.2:
        return "I'm not fully sure, but I think you are asking about " + intent
    return responses.get(intent, "Sorry, I didn’t understand your request.")

def text_to_speech(text, filename="output.mp3"):
    tts = gTTS(text=text)
    tts.save(filename)
    return filename

# ----------- ENDPOINTS -----------

@app.get("/")
def home():
    return {"message": "Voice Bot API is running!"}

# 🎤 FULL PIPELINE (MAIN ENDPOINT)
@app.post("/voicebot")
async def voicebot(file: UploadFile = File(...)):

    audio_path = "input_audio.m4a"

    # Save uploaded file
    with open(audio_path, "wb") as f:
        f.write(await file.read())

    try:
        # Speech → Text
        text = asr_model.transcribe(audio_path)["text"]

        # Intent
        intent, confidence = predict_intent(text)

        # Response
        response = generate_response(intent, confidence)

        # Text → Speech
        output_audio = text_to_speech(response)

        return {
            "transcription": text,
            "intent": intent,
            "confidence": float(confidence),
            "response": response,
            "audio_file": output_audio
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

# =========================================
# RUN SERVER
# =========================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)