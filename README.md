# 🤖 Production LLM Fine-Tuning - Complete Implementation

A production-grade implementation for fine-tuning Large Language Models (LLMs) for domain-specific applications. This project demonstrates end-to-end MLOps practices including data processing, model training with LoRA, evaluation, API deployment, and a user-friendly frontend.

## 🎯 Features

- **Efficient Fine-Tuning**: Uses LoRA (Low-Rank Adaptation) with 4-bit quantization for memory-efficient training
- **Production API**: FastAPI-based REST API for model inference
- **Interactive Frontend**: Streamlit-based web interface for easy interaction
- **RAG Support**: Retrieval-Augmented Generation pipeline for enhanced context
- **Comprehensive Evaluation**: ROUGE scores, exact match, and custom metrics
- **Docker Deployment**: Containerized setup for easy deployment
- **CI/CD Pipeline**: GitHub Actions for automated testing

## 📁 Project Structure

```
domain-llm-finetuning/
├── data/
│   ├── raw/                    # Original scraped data
│   ├── processed/              # Cleaned and formatted data
│   └── scripts/
│       ├── scraper.py         # Data collection
│       ├── preprocess.py      # Cleaning pipeline
│       └── create_dataset.py  # Format conversion
├── models/
│   ├── base/                   # Downloaded base models
│   └── finetuned/             # Your trained models
├── src/
│   ├── training/
│   │   ├── train.py           # Main training script
│   │   ├── config.py          # Training configuration
│   │   └── utils.py           # Helper functions
│   ├── evaluation/
│   │   ├── evaluate.py        # Model evaluation
│   │   └── metrics.py         # Custom metrics
│   ├── rag/
│   │   ├── vector_store.py    # ChromaDB setup
│   │   └── retrieval.py       # RAG pipeline
│   └── api/
│       ├── main.py            # FastAPI application
│       ├── models.py          # Pydantic schemas
│       └── inference.py       # Model inference
├── frontend/
│   └── app.py                 # Streamlit UI
├── tests/
│   ├── test_api.py
│   └── test_inference.py
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory analysis
│   ├── 02_training.ipynb      # Interactive training
│   └── 03_evaluation.ipynb    # Results visualization
├── configs/
│   ├── training_config.yaml
│   └── model_config.yaml
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .github/
│   └── workflows/
│       └── ci.yml             # CI/CD pipeline
├── requirements.txt
├── setup.py
└── README.md
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# Scrape data (or use your own data)
python data/scripts/scraper.py

# Preprocess data
python data/scripts/preprocess.py

# Create train/val/test splits
python data/scripts/create_dataset.py
```

### 3. Train Model

```bash
# Train the model
python src/training/train.py
```

**Note**: Training requires a GPU with at least 16GB VRAM. The model uses 4-bit quantization to reduce memory requirements.

### 4. Evaluate Model

```bash
# Run evaluation
python src/evaluation/evaluate.py
```

### 5. Run API Server

```bash
# Start FastAPI server
uvicorn src.api.main:app --reload
```

The API will be available at `http://localhost:8000`

### 6. Run Frontend

```bash
# Start Streamlit app (in a separate terminal)
streamlit run frontend/app.py
```

The frontend will be available at `http://localhost:8501`

### 7. Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose -f docker/docker-compose.yml up --build
```

## 📊 API Endpoints

### Health Check
```bash
GET /health
```

### Generate Text
```bash
POST /generate
Content-Type: application/json

{
  "prompt": "What are the symptoms of Type 2 diabetes?",
  "max_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9,
  "use_rag": false
}
```

## 🔧 Configuration

### Training Configuration

Edit `src/training/config.py` or `configs/training_config.yaml` to customize:

- Model selection
- LoRA parameters (r, alpha)
- Training hyperparameters (epochs, batch size, learning rate)
- Paths and output directories

### Model Configuration

Edit `configs/model_config.yaml` for:

- Model path
- Quantization settings
- Inference parameters

## 📈 Key Performance Indicators

### Training Metrics
- Loss curve convergence
- Perplexity: < 10 (domain-specific)
- Training time: 4-12 hours for 7B model

### Quality Metrics
- ROUGE-L: > 0.5
- Domain accuracy: > 80%
- Human evaluation: 4/5 rating

### Deployment Metrics
- Inference latency: < 500ms
- API uptime: > 99%
- Throughput: > 10 requests/sec

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html
```

## 📝 Development

### Code Formatting

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/
```

## 🐳 Docker

### Build Image

```bash
docker build -f docker/Dockerfile -t domain-llm-api .
```

### Run Container

```bash
docker run -p 8000:8000 --gpus all domain-llm-api
```

## 📚 Documentation

- [Training Guide](docs/training.md)
- [API Documentation](http://localhost:8000/docs) (when API is running)
- [Deployment Guide](docs/deployment.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face for transformers and datasets
- PEFT library for LoRA implementation
- FastAPI for the API framework
- Streamlit for the frontend framework

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Note**: This is a template project. Replace placeholder data and configurations with your actual domain-specific data and requirements.

