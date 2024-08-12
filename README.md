# EverythingAI: AI tasks trained without datasets.

![EverythingAI Logo](https://res.cloudinary.com/dqneghcme/image/upload/v1723446961/Imposingcontext_dvdhyf.jpg)

EverythingAI is a powerful API suite providing cutting-edge AI services powered by Google's Gemini model. Our platform stands out by implementing the innovative "Imposing Context" technique, significantly enhancing AI model performance and accuracy without requiring any specialized datasets.

## 🌟 Key Features

- 🎭 Deepfake Detection
- 🗣️ Speech-to-Text Conversion
- 🔍 Sentiment Analysis
- 📸 Image OCR
- 🤟 American Sign Language (ASL) Interpretation

## 🚀 Quick Start

Our API is live and ready for immediate use! You can start making requests right away using the examples below.

### API Base URL
```
https://test-xcrz.onrender.com
```

## 🧠 Context Injection Technique

MultiAI leverages the Context Injection technique to optimize Gemini model performance:

1. Creating simulated chat history to "train" the model.
2. Providing specific instructions and examples for each task.
3. Optimizing context for accurate and consistent results.

Benefits:
- Improved accuracy
- Reduced processing time
- Enhanced result consistency

## 🛠 API Usage Examples

### 1. Deepfake Detection

```javascript
async function callDeepfakeDetectionAPI(videoUrl) {
  try {
    const response = await fetch(`https://test-xcrz.onrender.com/detect_deepfake?file_url=${videoUrl}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json'
      }
    });
    if (!response.ok) {
      throw new Error(`API request failed with status ${response.status}`);
    }
    const data = await response.json();
    return data.message;
  } catch (error) {
    console.error('Error calling Deepfake Detection API:', error);
    throw error;
  }
}

// Example usage
const videoUrl = 'https://example/video.mp4';
callDeepfakeDetectionAPI(videoUrl)
  .then(result => {
    console.log('Deepfake Detection Result:', result);
  })
  .catch(error => {
    console.error('Error:', error);
  });
```

### 2. Speech-to-Text Conversion

```javascript
async function callSpeechToTextAPI(audioUrl, language = 'English') {
  try {
    const response = await fetch(`https://test-xcrz.onrender.com/speech_to_text?file_url=${audioUrl}&language=${language}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json'
      }
    });
    if (!response.ok) {
      throw new Error(`API request failed with status ${response.status}`);
    }
    const data = await response.json();
    return data.message;
  } catch (error) {
    console.error('Error calling Speech-to-Text API:', error);
    throw error;
  }
}

// Example usage
const audioUrl = 'https://example.com/audio.mp3';
callSpeechToTextAPI(audioUrl, 'English')
  .then(result => {
    console.log('Speech-to-Text Result:', result);
  })
  .catch(error => {
    console.error('Error:', error);
  });
```

### 3. Sentiment Analysis

```javascript
async function callSentimentAnalysisAPI(text) {
  try {
    const response = await fetch(`https://test-xcrz.onrender.com/analyze_sentiment?text=${text}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json'
      }
    });
    if (!response.ok) {
      throw new Error(`API request failed with status ${response.status}`);
    }
    const data = await response.json();
    return data.sentiment;
  } catch (error) {
    console.error('Error calling Sentiment Analysis API:', error);
    throw error;
  }
}

// Example usage
const textToAnalyze = 'I love this product!';
callSentimentAnalysisAPI(textToAnalyze)
  .then(result => {
    console.log('Sentiment Analysis Result:', result);
  })
  .catch(error => {
    console.error('Error:', error);
  });
```

### 4. Image OCR

```javascript
async function callImageOCRAPI(imageUrl, language = 'English') {
  try {
    const response = await fetch(`https://test-xcrz.onrender.com/ocr_image?url=${imageUrl}&language=${language}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
    });
    if (!response.ok) {
      throw new Error(`API request failed with status ${response.status}`);
    }
    const data = await response.json();
    return data.ocr_result;
  } catch (error) {
    console.error('Error calling Image OCR API:', error);
    throw error;
  }
}

// Example usage
const imageUrl = 'https://example.com/image.png';
language = 'English'

callImageOCRAPI(imageUrl, language)
  .then(result => {
    console.log('Image OCR Result:', result);
  })
  .catch(error => {
    console.error('Error:', error);
  });
```

### 5. ASL Interpretation

```javascript
async function callASLInterpreterAPI(videoUrl) {
  try {
    const response = await fetch(`https://test-xcrz.onrender.com/interpret_asl?url=${videoUrl}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      }
    });

    if (!response.ok) {
      throw new Error(`API request failed with status ${response.status}`);
    }

    const data = await response.json();
    return data.asl_interpretation;

  } catch (error) {
    console.error('Error calling ASL Interpreter API:', error);
    throw error;
  }
}

// Example usage
const videoUrl = 'https://example/video.mp4'; 

callASLInterpreterAPI(videoUrl)
  .then(interpretation => {
    console.log('ASL Interpretation:', interpretation); 
  })
  .catch(error => {
    console.error('Error:', error);
  });

```

## 🔧 Local Development

To set up the project locally for development or customization:

1. Clone the repository:
   ```
   git clone https://github.com/your-username/multiai.git
   ```
2. Navigate to the project directory:
   ```
   cd multiai
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Start the local server:
   ```
   uvicorn main:app --reload
   ```

## 🤝 Contributing

We welcome contributions from the community! Please see our [Contribution Guidelines](CONTRIBUTING.md) for more details on how to get involved.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 📞 Contact

For any inquiries or support, please contact our team at [support@multiai.com](mailto:support@multiai.com).

---

⭐️ If you find MultiAI useful, don't forget to give it a star on GitHub!

```
