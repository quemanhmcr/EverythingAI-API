import os
import time
from typing import List, Optional
import google.generativeai as genai
import requests
from fastapi import FastAPI, HTTPException

app = FastAPI()

# Configuration
MODEL_NAME = "gemini-1.5-flash"
GENERATION_CONFIG = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Initialize Gemini API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel(model_name=MODEL_NAME, generation_config=GENERATION_CONFIG)

class FileProcessor:
    @staticmethod
    def upload_to_gemini(path: str, mime_type: Optional[str] = None):
        file = genai.upload_file(path, mime_type=mime_type)
        print(f"Uploaded file '{file.display_name}' as: {file.uri}")
        return file

    @staticmethod
    def wait_for_files_active(files) -> None:
        print("Waiting for file processing...")
        for file in files:
            while file.state.name == "PROCESSING":
                print(".", end="", flush=True)
                time.sleep(10)
                file = genai.get_file(file.name)
            if file.state.name != "ACTIVE":
                raise Exception(f"File {file.name} failed to process")
        print("\n...all files ready")

class VideoDownloader:
    @staticmethod
    def download_video(url: str, filename: str) -> None:
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded video to {filename}")
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Error downloading video: {str(e)}")

class DeepfakeDetector:
    TRAINING_SOURCES = [
        "https://res.cloudinary.com/dqneghcme/video/upload/v1722939258/Y2meta.app-This_is_not_Morgan_Freeman_-_A_Deepfake_Singularity-_1080p_veegqk.mp4",
        "https://res.cloudinary.com/dqneghcme/video/upload/v1722931488/9_xibnhz.mp4",
        "https://res.cloudinary.com/dqneghcme/video/upload/v1722931436/lv_0_20240802211151_ll0hvt.mp4",
    ]

    def __init__(self):
        self.training_files = self._prepare_training_files()

    def _prepare_training_files(self):
        files = []
        for i, url in enumerate(self.TRAINING_SOURCES):
            filename = f"video_{i+1}.mp4"
            VideoDownloader.download_video(url, filename)
            files.append(FileProcessor.upload_to_gemini(filename, mime_type="video/mp4"))
        FileProcessor.wait_for_files_active(files)
        return files

    def detect(self, url_video: str) -> str:
        filename = "detect_video.mp4"
        VideoDownloader.download_video(url_video, filename)
        
        file_to_detect = FileProcessor.upload_to_gemini(filename, mime_type="video/mp4")
        FileProcessor.wait_for_files_active([file_to_detect])
        
        chat_session = model.start_chat(
            history=[
                {"role": "user", "parts": [self.training_files[0]]},
                {"role": "model", "parts": ["Video này là một video được tạo ra bởi deepfake. Dù nó rất tinh vi nhưng tôi có thể nhìn ra được"]},
                {"role": "user", "parts": [self.training_files[1]]},
                {"role": "user", "parts": ["Đây là video deepfake hay người thật"]},
                {"role": "model", "parts": ["Đây là một video deepfake, dù tinh vi nhưng tôi có thể nhận ra được. Người thật có thể dễ dàng nhận thấy hơn"]},
                {"role": "user", "parts": [file_to_detect]},
            ]
        )
        
        response = chat_session.send_message("Phản hồi nhị phân, Đây là video deepfake hay người thật.")
        os.remove(filename)
        return response.text

class SpeechToTextConverter:
    TRAINING_AUDIO = ["https://res.cloudinary.com/dqneghcme/video/upload/v1723244157/dsgt_eudonh.mp3"]
    TRAINING_RESULT = "Welcome to the Gemini API developer competition, where your app could land you bragging rights with a real trophy, cash prizes and for the best app, an electric-powered 1981 classic Delorean. [...]"

    def __init__(self):
        self.training_files = self._prepare_training_files()

    def _prepare_training_files(self):
        files = []
        for i, url in enumerate(self.TRAINING_AUDIO):
            filename = f"audio_{i+1}.mp3"
            VideoDownloader.download_video(url, filename)
            files.append(FileProcessor.upload_to_gemini(filename, mime_type="audio/mpeg"))
        FileProcessor.wait_for_files_active(files)
        return files

    def convert(self, url_audio: str, language: str = "English") -> str:
        filename = f"audio_detect_{len(self.training_files)}.mp3"
        VideoDownloader.download_video(url_audio, filename)

        file_to_convert = FileProcessor.upload_to_gemini(filename, mime_type="audio/mpeg")
        FileProcessor.wait_for_files_active([file_to_convert])

        chat_session = model.start_chat(
              history=[
                  {"role": "user", "parts": [self.training_files[0]]},
                  {"role": "user", "parts": ["Transcribe this audio to text in English:"]},
                  {"role": "model", "parts": [self.TRAINING_RESULT]},
                  {"role": "user", "parts": ["Great job. Now, I need you to follow these guidelines for all future transcriptions:"]},
                  {"role": "model", "parts": ["Understood. I'm ready to follow your guidelines for future transcriptions."]},
                  {"role": "user", "parts": [
                      "1. Focus only on human speech. Ignore background noise or music unless it's relevant to understanding the speech.",
                      "2. If a part of the audio is unclear or you're not confident about what was said, indicate this by writing [unclear] or [inaudible].",
                      "3. If there are long pauses or significant non-speech parts, note them as [pause] or [background noise].",
                      "4. Use proper punctuation and capitalization to make the transcript readable.",
                      "5. Include speaker labels if multiple speakers are present, e.g., [Speaker 1], [Speaker 2].",
                      "6. If the audio is in a language other than the one requested, note this at the beginning of the transcript.",
                      "7. Keep your response concise. Don't explain your process or add unnecessary commentary."
                  ]},
                  {"role": "model", "parts": ["I understand and will follow these guidelines for all transcriptions. I'm ready to proceed with the next audio file."]},
                  {"role": "user", "parts": [file_to_convert]},
              ]
          )

        response = chat_session.send_message(f"Transcribe this audio to text in {language}. Follow the guidelines provided earlier:")
        os.remove(filename)
        return response.text

class SentimentAnalyzer:
    def __init__(self):
        # Khởi tạo nếu cần
        pass

    def analyze(self, text: str) -> str:
        chat_session = model.start_chat(
            history=[
                {"role": "user", "parts": ["Analyze the sentiment of the following text: 'I love this product!'"]},
                {"role": "model", "parts": ["The sentiment of the text 'I love this product!' is happy. The use of the word 'love' expresses a strong positive emotion towards the product."]},
                {"role": "user", "parts": ["Analyze the sentiment of the following text: 'This is the worst experience ever.'"]},
                {"role": "model", "parts": ["The sentiment of the text 'This is the worst experience ever.' is negative. The use of the word 'worst' indicates a very negative perception of the experience."]},
            ]
        )

        response = chat_session.send_message(f"Analyze the sentiment of the following text: '{text}'. Respond with just one of in 80 human emotional states")
        return response.text.strip()


class ImageOCR:
    def __init__(self):
        self.file_processor = FileProcessor()
        self.video_downloader = VideoDownloader()

    def get_file_extension(self, url: str) -> str:
        """Gets the file extension from the given URL."""
        last_slash = url.rfind('/')
        if last_slash == -1:
            return ""
        
        filename = url[last_slash + 1:]
        last_dot = filename.rfind('.')
        if last_dot == -1:
            return ""
        
        return filename[last_dot:].lower()

    def process_image(self, url: str, language: str) -> str:
        """Processes an image from the given URL and performs OCR."""
        extension = self.get_file_extension(url)
        if extension not in ['.png', '.jpg', '.jpeg']:
            return "Unsupported file format."

        filename = f"temp_image_{int(time.time())}{extension}"
        
        try:
            self.video_downloader.download_video(url, filename)
            
            mime_type = "image/png" if extension == '.png' else "image/jpeg"
            file = self.file_processor.upload_to_gemini(filename, mime_type=mime_type)
            
            if not file:
                return "Failed to upload the image to Gemini."

            self.file_processor.wait_for_files_active([file])

            chat_session = model.start_chat(history=[
                {"role": "user", "parts": [file]},
                {"role": "user", "parts": [f"OCR {language} this image:"]},
            ])

            response = chat_session.send_message(f"OCR {language} this image:")
            return response.text

        except Exception as e:
            return f"Error during OCR processing: {e}"
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class ASLInterpreter:
    def __init__(self):
        pass

    def interpret(self, url_video: str) -> str:
        filename = "asl_video_to_interpret.mp4"
        VideoDownloader.download_video(url_video, filename)

        file_to_interpret = FileProcessor.upload_to_gemini(filename, mime_type="video/mp4")
        FileProcessor.wait_for_files_active([file_to_interpret])

        chat_session = model.start_chat(
            history=[
                {"role": "user", "parts": ["You are an expert ASL interpreter. Your task is to watch videos of people using American Sign Language and provide accurate, detailed translations into written English."]},
                {"role": "model", "parts": ["Understood. I am an expert ASL interpreter capable of watching videos of American Sign Language and providing accurate, detailed translations into written English. I will focus on the hand movements, facial expressions, and body language to interpret the signs and convey the full meaning of the communication."]},
                {"role": "user", "parts": ["When interpreting ASL videos, please follow these guidelines:\n1. Pay close attention to hand shapes, movements, and positions.\n2. Note facial expressions and body language, as they are crucial for conveying emotion and meaning in ASL.\n3. Interpret not just individual signs, but the overall context and meaning of the communication.\n4. Provide a natural, fluent English translation that captures the intent and tone of the signer.\n5. If there are any signs or concepts that are unclear or ambiguous, mention this in your interpretation.\n6. Be sensitive to and accurately convey any cultural nuances specific to the Deaf community."]},
                {"role": "model", "parts": ["Thank you for the detailed guidelines. I will adhere to them when interpreting ASL videos. I'm ready to interpret ASL videos with these principles in mind."]},
                {"role": "user", "parts": [file_to_interpret]},
            ]
        )


        response = chat_session.send_message("Please provide a detailed English translation of the ASL communication in this video, following the guidelines we discussed:")
        os.remove(filename)
        return response.text


# Initialize services
deepfake_detector = DeepfakeDetector()
speech_to_text_converter = SpeechToTextConverter()
sentiment_analyzer = SentimentAnalyzer()
image_ocr = ImageOCR()
asl_interpreter = ASLInterpreter()


@app.get("/detect_deepfake")
def detect_deepfake(file_url: str):
    try:
        result = deepfake_detector.detect(file_url)
        return {"message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/speech_to_text")
def speech_to_text(file_url: str, language: str = "English"):
    try:
        result = speech_to_text_converter.convert(file_url, language)
        return {"message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyze_sentiment")
def analyze_sentiment(text: str):
    try:
        result = sentiment_analyzer.analyze(text)
        return {"sentiment": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ocr_image")
async def ocr_image(url: str, language: str = "English"):
    try:
        result = image_ocr.process_image(url, language)
        return {"ocr_result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/interpret_asl")
async def interpret_asl(url: str):
    try:
        result = asl_interpreter.interpret(url)
        return {"asl_interpretation": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
