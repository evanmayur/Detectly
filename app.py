import logging
import hashlib
import os
import json
import io
import time
import pickle
import re
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from scipy.stats import norm

from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
import PyPDF2
import docx
import redis
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
import textstat

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('ai_detector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['APPLICATION_ROOT'] = '/api'

# Configure CORS
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Configuration
class Config:
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_TEXT_LENGTH = 10000
    CACHE_TTL = 3600  # 1 hour
    MIN_TEXT_LENGTH = 50  # Minimum characters for reliable detection
    MODEL_PATH = 'detector_model_20250702_100950.pkl'
    VALIDATION_DATA_PATH = 'validation_results_20250702_100950.csv'

config = Config()

# Initialize Redis
try:
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        db=int(os.getenv('REDIS_DB', 0)),
        decode_responses=True,
        socket_timeout=5,
        socket_connect_timeout=5
    )
    redis_client.ping()
    USE_REDIS = True
    logger.info("Redis connection established")
except Exception as e:
    USE_REDIS = False
    logger.warning(f"Redis not available, using in-memory cache: {str(e)}")

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://" if not USE_REDIS else f"redis://{redis_client.connection_pool.connection_kwargs['host']}:{redis_client.connection_pool.connection_kwargs['port']}"
)

class HybridAIDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load GPT-2 for perplexity calculation
        self.gpt2_model_name = "gpt2-medium"
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(self.gpt2_model_name)
        self.gpt2_model = GPT2LMHeadModel.from_pretrained(self.gpt2_model_name).to(self.device)
        self.gpt2_model.eval()
        
        # Load validation data for confidence calibration
        self.validation_data = self.load_validation_data()
        self.human_mean, self.human_std = self.calculate_human_stats()
        
        logger.info("Hybrid AI detection model loaded successfully")

    def load_validation_data(self):
        """Load validation data for confidence calibration"""
        try:
            df = pd.read_csv(config.VALIDATION_DATA_PATH)
            logger.info(f"Validation data loaded with {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Error loading validation data: {e}")
            return pd.DataFrame()

    def calculate_human_stats(self):
        """Calculate statistics for human-written text from validation data"""
        if not self.validation_data.empty:
            human_data = self.validation_data[self.validation_data['true_label'] == 'human']
            if not human_data.empty:
                mean = human_data['perplexity'].mean()
                std = human_data['perplexity'].std()
                logger.info(f"Human text stats - Mean: {mean:.2f}, Std: {std:.2f}")
                return mean, std
        return 45.0, 15.0  # Default values if no data

    def calculate_perplexity(self, text: str) -> float:
        """Calculate perplexity of the given text"""
        encodings = self.gpt2_tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        input_ids = encodings.input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = self.gpt2_model(input_ids, labels=input_ids)
            loss = outputs.loss
        
        return torch.exp(loss).item()

    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """Extract comprehensive linguistic features for detection"""
        features = {}
        try:
            # Basic text statistics
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            word_count = len(words)
            
            # Sentence statistics
            if sentences:
                sent_lengths = [len(word_tokenize(sent)) for sent in sentences]
                features['sentences'] = len(sentences)
                features['avg_sentence_len'] = np.mean(sent_lengths)
                features['sentence_variation'] = np.std(sent_lengths) / np.mean(sent_lengths) if len(sentences) > 1 else 0
            else:
                features['sentences'] = 0
                features['avg_sentence_len'] = 0
                features['sentence_variation'] = 0
            
            # Hedging phrases detection
            hedging_phrases = [
                "it is important", "it is worth noting", "however", "it is crucial",
                "it is essential", "we must consider", "we must take into account",
                "in summary", "in conclusion", "this suggests that"
            ]
            features['hedging_phrases'] = sum(
                1 for phrase in hedging_phrases if re.search(r'\b' + re.escape(phrase), text, re.IGNORECASE)
            )
            
            # Lexical diversity
            if words:
                unique_words = set(words)
                features['vocab_richness'] = len(unique_words) / word_count  # TTR
                features['avg_word_len'] = sum(len(word) for word in words) / word_count
            else:
                features['vocab_richness'] = 0
                features['avg_word_len'] = 0
                
            # Readability scores
            features['flesch_score'] = textstat.flesch_reading_ease(text)
            features['gunning_fog'] = textstat.gunning_fog(text)
            features['smog_index'] = textstat.smog_index(text)
            
            # Repeated phrases detection
            repeated_phrase_count = 0
            if words and len(words) > 3:
                # Detect consecutive 3-word repetitions
                trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
                trigram_counts = {}
                for trigram in trigrams:
                    trigram_counts[trigram] = trigram_counts.get(trigram, 0) + 1
                repeated_phrase_count = sum(count > 1 for count in trigram_counts.values())
            features['repeated_phrases'] = repeated_phrase_count
            
            # Punctuation analysis
            features['comma_count'] = text.count(',')
            features['question_count'] = text.count('?')
            features['exclamation_count'] = text.count('!')
            
            # Part-of-speech analysis
            if words:
                pos_tags = pos_tag(words)
                # Count different POS categories
                noun_count = sum(1 for _, tag in pos_tags if tag.startswith('NN'))
                verb_count = sum(1 for _, tag in pos_tags if tag.startswith('VB'))
                adj_count = sum(1 for _, tag in pos_tags if tag.startswith('JJ'))
                adv_count = sum(1 for _, tag in pos_tags if tag.startswith('RB'))
                pronoun_count = sum(1 for _, tag in pos_tags if tag.startswith('PRP'))
                
                features['noun_ratio'] = noun_count / word_count
                features['verb_ratio'] = verb_count / word_count
                features['adj_ratio'] = adj_count / word_count
                features['adv_ratio'] = adv_count / word_count
                features['pronoun_ratio'] = pronoun_count / word_count
            else:
                features['noun_ratio'] = 0
                features['verb_ratio'] = 0
                features['adj_ratio'] = 0
                features['adv_ratio'] = 0
                features['pronoun_ratio'] = 0
                
            # Syntactic complexity
            if sentences:
                complex_sentence_count = 0
                for sent in sentences:
                    if len(word_tokenize(sent)) > 20 and (',' in sent or ';' in sent):
                        complex_sentence_count += 1
                features['complex_sentence_ratio'] = complex_sentence_count / len(sentences)
            else:
                features['complex_sentence_ratio'] = 0
                
            # Unique bigrams ratio
            if words and len(words) > 1:
                bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
                unique_bigrams = set(bigrams)
                features['bigram_diversity'] = len(unique_bigrams) / len(bigrams)
            else:
                features['bigram_diversity'] = 0
                
            # Personal expression features
            features['first_person'] = text.lower().count(' i ') + text.lower().count(' we ') + text.lower().count(' my ')
            features['second_person'] = text.lower().count(' you ') + text.lower().count(' your ')
            
        except Exception as e:
            logger.error(f"Error extracting linguistic features: {e}")
            # Set default values for all features
            features = {
                'sentences': 0, 'avg_sentence_len': 0, 'sentence_variation': 0,
                'hedging_phrases': 0, 'vocab_richness': 0, 'avg_word_len': 0,
                'flesch_score': 0, 'gunning_fog': 0, 'smog_index': 0,
                'repeated_phrases': 0, 'comma_count': 0, 'question_count': 0,
                'exclamation_count': 0, 'noun_ratio': 0, 'verb_ratio': 0,
                'adj_ratio': 0, 'adv_ratio': 0, 'pronoun_ratio': 0,
                'complex_sentence_ratio': 0, 'bigram_diversity': 0,
                'first_person': 0, 'second_person': 0
            }
        
        return features

    def detect_ai_text(self, text: str) -> Dict:
        """
        Detect AI-generated text using hybrid approach
        Returns comprehensive detection results, mapped to frontend expectations
        """
        if len(text) < config.MIN_TEXT_LENGTH:
            return {
                "error": f"Text too short (minimum {config.MIN_TEXT_LENGTH} characters required)",
                "success": False
            }

        start_time = time.time()
        try:
            # Calculate perplexity
            perplexity = self.calculate_perplexity(text)

            # Extract linguistic features
            linguistic_features = self.extract_linguistic_features(text)

            # Calculate base AI probability
            z_score = (perplexity - self.human_mean) / self.human_std
            base_ai_prob = 1 - norm.cdf(z_score)
            
            # Feature-based adjustments for human-like text
            human_likeness = 0
            
            # High TTR increases human likelihood
            if linguistic_features['vocab_richness'] > 0.65:
                human_likeness += 0.15
                
            # High sentence variation increases human likelihood
            if linguistic_features['sentence_variation'] > 0.5:
                human_likeness += 0.10
                
            # High readability score increases human likelihood
            if linguistic_features['flesch_score'] > 60:
                human_likeness += 0.10
                
            # Repeated phrases increase human likelihood (within reason)
            if 1 < linguistic_features['repeated_phrases'] < 10:
                human_likeness += min(0.05, linguistic_features['repeated_phrases'] * 0.005)
                
            # Hedging phrases increase human likelihood
            if linguistic_features['hedging_phrases'] > 2:
                human_likeness += min(0.15, linguistic_features['hedging_phrases'] * 0.05)
                
            # High bigram diversity indicates human writing
            if linguistic_features['bigram_diversity'] > 0.85:
                human_likeness += 0.10
                
            # Complex sentence structures indicate human writing
            if linguistic_features['complex_sentence_ratio'] > 0.3:
                human_likeness += 0.10
                
            # Personal expressions indicate human writing
            if linguistic_features['first_person'] > 3 or linguistic_features['second_person'] > 3:
                human_likeness += 0.10
                
            # Punctuation diversity indicates human writing
            punctuation_diversity = (
                linguistic_features['comma_count'] + 
                linguistic_features['question_count'] + 
                linguistic_features['exclamation_count']
            )
            if punctuation_diversity > 5:
                human_likeness += min(0.10, punctuation_diversity * 0.02)
                
            # Apply human likeness bonus (capped at 60% reduction)
            ai_prob = base_ai_prob * (1 - min(0.6, human_likeness))

            # Determine if text is likely AI-generated with higher threshold
            threshold = 0.85  # Increased to reduce false positives
            is_ai = ai_prob > threshold

            # Confidence: distance from threshold, with smoothing
            if is_ai:
                confidence = (ai_prob - threshold) / (1.0 - threshold)
            else:
                confidence = (threshold - ai_prob) / threshold
                
            # Smooth confidence to avoid extreme values (5%-100% range)
            confidence = min(1.0, max(0.05, confidence))

            # Model scores breakdown
            model_scores = {
                "perplexity": round(base_ai_prob * 100, 2),
                "linguistic": round(human_likeness * 100, 2)
            }

            # Verdict
            verdict = "ai" if is_ai else "human"

            # Word count
            word_count = len(text.split())

            # Compose response for frontend
            response = {
                "ai_score": round(ai_prob * 100, 2),
                "confidence": confidence,
                "model_scores": model_scores,
                "features": linguistic_features,
                "processing_time": round(time.time() - start_time, 3),
                "word_count": word_count,
                "verdict": verdict,
                "success": True
            }

            return response

        except Exception as e:
            logger.error(f"Detection error: {e}")
            return {
                "error": "Error processing text",
                "success": False
            }

# Initialize the detector
detector = HybridAIDetector()

def cache_key(text: str) -> str:
    """Generate a cache key for the given text"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def get_cached_result(key: str) -> Optional[Dict]:
    """Get cached result if available"""
    if not USE_REDIS:
        return None
    
    try:
        cached = redis_client.get(f"ai_detect:{key}")
        return json.loads(cached) if cached else None
    except Exception as e:
        logger.warning(f"Redis get error: {e}")
        return None

def set_cached_result(key: str, result: Dict, ttl: int = config.CACHE_TTL) -> None:
    """Cache the result with a TTL"""
    if not USE_REDIS:
        return
    
    try:
        redis_client.setex(
            f"ai_detect:{key}",
            ttl,
            json.dumps(result)
        )
    except Exception as e:
        logger.warning(f"Redis set error: {e}")

def extract_text_from_file(file) -> str:
    """Extract text from various file formats"""
    try:
        filename = file.filename.lower()
        
        if filename.endswith('.pdf'):
            reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
            text = "\n".join([page.extract_text() for page in reader.pages])
        elif filename.endswith('.docx'):
            doc = docx.Document(io.BytesIO(file.read()))
            text = "\n".join([para.text for para in doc.paragraphs])
        elif filename.endswith(('.txt', '.md')):
            text = file.read().decode('utf-8', errors='replace')
        else:
            raise ValueError("Unsupported file format")
            
        return text[:config.MAX_TEXT_LENGTH]
    
    except Exception as e:
        logger.error(f"Error extracting text from file: {e}")
        raise ValueError(f"Could not extract text from file: {str(e)}")

def validate_file(file) -> None:
    """Validate the uploaded file"""
    if not file:
        raise ValueError("No file uploaded")
    
    if file.content_length > config.MAX_FILE_SIZE:
        raise ValueError(f"File size exceeds {config.MAX_FILE_SIZE/1024/1024}MB limit")
    
    filename = file.filename.lower()
    if not filename.endswith(('.pdf', '.docx', '.txt', '.md')):
        raise ValueError("Only PDF, DOCX, TXT, and MD files are supported")

@app.route('/api/analyze', methods=['POST'])
@limiter.limit("10 per minute")
def analyze_text():
    """Endpoint for text analysis"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text or len(text) < config.MIN_TEXT_LENGTH:
            return jsonify({
                'error': f'Text too short (minimum {config.MIN_TEXT_LENGTH} characters required)',
                'success': False
            }), 400
            
        # Check cache
        cache_key_val = cache_key(text)
        cached_result = get_cached_result(cache_key_val)
        if cached_result:
            cached_result['is_cached'] = True
            return jsonify(cached_result)

        # Process the text
        result = detector.detect_ai_text(text)

        if result.get('success', False):
            result['is_cached'] = False
            set_cached_result(cache_key_val, result)

        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return jsonify({
            'error': 'Internal server error',
            'success': False
        }), 500

@app.route('/api/upload', methods=['POST'])
@limiter.limit("5 per minute")
def upload_file():
    """Endpoint for file upload and analysis"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file uploaded',
                'success': False
            }), 400
        
        file = request.files['file']
        validate_file(file)
        
        text = extract_text_from_file(file)
        cache_key_val = cache_key(text)
        
        # Check cache
        cached_result = get_cached_result(cache_key_val)
        if cached_result:
            cached_result['is_cached'] = True
            return jsonify(cached_result)

        # Process the text
        result = detector.detect_ai_text(text)

        if result.get('success', False):
            result['is_cached'] = False
            set_cached_result(cache_key_val, result)

        return jsonify(result)
    
    except ValueError as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 400
    except Exception as e:
        logger.error(f"File detection error: {e}")
        return jsonify({
            'error': 'Internal server error',
            'success': False
        }), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    try:
        # Test model availability
        test_text = "This is a health check of the AI detection system."
        result = detector.detect_ai_text(test_text)
        if not result.get('success', False):
            raise RuntimeError("Model test failed")
        
        # Test Redis if available
        if USE_REDIS:
            redis_client.ping()
        
        return jsonify({
            'status': 'healthy',
            'model': 'hybrid-20250702',
            'human_text_stats': {
                'mean': detector.human_mean,
                'std': detector.human_std
            },
            'redis_available': USE_REDIS,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.errorhandler(429)
def ratelimit_handler(e):
    """Custom rate limit exceeded response"""
    return jsonify({
        'error': f'Rate limit exceeded: {e.description}',
        'success': False
    }), 429

if __name__ == '__main__':
    # Validate models before starting
    try:
        logger.info("Validating models...")
        test_text = "This is a test sentence for startup validation."
        detector.detect_ai_text(test_text)
        logger.info("Models validated successfully")
    except Exception as e:
        logger.critical(f"Failed to validate models: {e}")
        raise
    
    logger.info("Starting AI Detection Server")
    app.run(host='0.0.0.0', port=5001, threaded=True)