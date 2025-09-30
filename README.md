# Dynamic AI Chatbot - Emotion and Sentiment Analysis

## Project Overview

This project implements a **Dynamic AI Chatbot** with advanced emotion and sentiment analysis capabilities using state-of-the-art Natural Language Processing (NLP) and Deep Learning techniques. The system can understand user emotions and sentiments from text input, enabling more empathetic and contextually appropriate responses.

## Key Features

### 🎭 Emotion Classification
- **6 Emotion Categories**: joy, sadness, anger, fear, love, surprise
- **High Accuracy Model**: 90.77% validation accuracy
- **Real-time Analysis**: Instant emotion detection from text input

### 😊 Sentiment Analysis  
- **3 Sentiment Categories**: positive, negative, neutral
- **Dual Classification**: Comprehensive sentiment understanding
- **Context-Aware**: Maintains conversation context

### 🤖 AI-Powered Architecture
- **BERT-based Models**: Fine-tuned transformer models for optimal performance
- **Multi-Task Learning**: Simultaneous emotion and sentiment analysis
- **Scalable Design**: Can handle large volumes of text data

## Technical Implementation

### Models Used
- **Emotion Classification**: Fine-tuned BERT-tiny model
- **Sentiment Analysis**: Custom-trained classifier
- **Dataset**: Combined emotion (422K+ samples) and sentiment (3.3K+ samples) datasets

### Performance Metrics
```
🎭 Emotion Model Accuracy: 90.77%
📊 Detailed Classification Report:
- joy: 95% precision, 92% recall
- sadness: 94% precision, 95% recall  
- anger: 92% precision, 89% recall
- fear: 86% precision, 85% recall
- love: 80% precision, 86% recall
- surprise: 67% precision, 85% recall
```

## Project Structure

```
dynamic-ai-chatbot/
├── models/                    # Saved model files
├── results/                   # Training results and metrics
├── data/                      # Dataset files
│   ├── combined_emotion.csv
│   └── combined_sentiment_data.csv
├── notebooks/                 # Jupyter notebooks
│   └── max.ipynb             # Main training notebook
├── requirements.txt          # Python dependencies
└── README.md                # Project documentation
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- Transformers library
- CUDA-capable GPU (recommended)

### Installation Steps
1. Clone the repository:
```bash
git clone <repository-url>
cd dynamic-ai-chatbot
```

2. Install required packages:
```bash
pip install torch transformers pandas scikit-learn matplotlib seaborn accelerate joblib
```

3. Prepare your datasets:
   - Place `combined_emotion.csv` and `combined_sentiment_data.csv` in the project root
   - Or use the built-in sample data generator

## Usage

### Training the Models
```python
# Run the training notebook
jupyter notebook max.ipynb
```

The notebook includes:
- Data exploration and visualization
- Model training with BERT-tiny
- Performance evaluation
- Model saving and export

### Real-time Inference
```python
from chatbot_inference import EmotionSentimentAnalyzer

# Initialize the analyzer
analyzer = EmotionSentimentAnalyzer()

# Analyze text
text = "I'm feeling absolutely wonderful today!"
result = analyzer.analyze(text)

print(f"Emotion: {result['emotion']}")
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']}")
```

## API Integration

### Web Framework Integration
```python
from flask import Flask, request, jsonify
from chatbot_inference import EmotionSentimentAnalyzer

app = Flask(__name__)
analyzer = EmotionSentimentAnalyzer()

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.json
    text = data.get('text', '')
    result = analyzer.analyze(text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
```

### Response Generation
```python
def generate_response(emotion, sentiment, user_input):
    """
    Generate contextual responses based on detected emotion and sentiment
    """
    emotion_responses = {
        'joy': "I'm glad you're feeling happy!",
        'sadness': "I'm here to help if you're feeling down.",
        'anger': "I understand you're upset. Let's work through this.",
        'fear': "It's okay to feel scared sometimes.",
        'love': "That's wonderful to hear!",
        'surprise': "Wow, that's surprising!"
    }
    
    sentiment_modifiers = {
        'positive': "That sounds great! ",
        'negative': "I'm sorry to hear that. ",
        'neutral': "I see. "
    }
    
    base_response = emotion_responses.get(emotion, "Thank you for sharing.")
    modifier = sentiment_modifiers.get(sentiment, "")
    
    return modifier + base_response
```

## Dataset Information

Link : https://www.kaggle.com/datasets/kushagra3204/sentiment-and-emotion-analysis-dataset
### Emotion Dataset
- **Total Samples**: 422,746
- **Classes**: joy, sad, anger, fear, love, suprise
- **Distribution**: Balanced across emotions with joy and sadness being most common

### Sentiment Dataset  
- **Total Samples**: 3,309
- **Classes**: positive, negative
- **Distribution**: Nearly balanced between positive and negative

## Model Architecture

### Emotion Classification Model
- **Base Model**: BERT-tiny (prajjwal1/bert-tiny)
- **Training Epochs**: 3
- **Batch Size**: 32 (training), 64 (validation)
- **Sequence Length**: 128 tokens
- **Optimizer**: AdamW with default parameters

### Key Technical Features
- **Transfer Learning**: Leverages pre-trained BERT embeddings
- **Fine-tuning**: Custom classification heads for specific tasks
- **Multi-label Support**: Can be extended for multi-label classification
- **Efficient Inference**: Optimized for real-time applications

## Deployment Options

### 1. Local Deployment
```bash
python app.py
```

### 2. Docker Deployment
```dockerfile
FROM python:3.8-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]
```

### 3. Cloud Deployment
- AWS SageMaker for model hosting
- Google Cloud AI Platform
- Azure Machine Learning

## Performance Optimization

### Model Optimization
- Quantization for faster inference
- ONNX conversion for cross-platform compatibility
- Model pruning for reduced size

### Caching Strategy
- Redis for frequent query caching
- Response caching for common patterns
- Session management for conversation context

## Future Enhancements

### Planned Features
1. **Multilingual Support**
   - Extend to multiple languages
   - Cross-lingual transfer learning

2. **Voice Integration**
   - Speech-to-text conversion
   - Text-to-speech responses

3. **Advanced Analytics**
   - Conversation flow analysis
   - User behavior tracking
   - Performance monitoring dashboard

4. **Enhanced Models**
   - Larger transformer models (BERT-base, RoBERTa)
   - Ensemble methods for improved accuracy
   - Domain-specific fine-tuning

## Contributing

We welcome contributions! Please see our contributing guidelines for details on:
- Code style and standards
- Testing requirements
- Pull request process

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue in the GitHub repository
- Contact the development team
- Check the documentation and examples

## Acknowledgments

- Hugging Face for transformer models and libraries
- PyTorch team for deep learning framework
- Dataset contributors for emotion and sentiment data

---

**Note**: This is a dynamic project with continuous improvements. Check the repository for the latest updates and features.
