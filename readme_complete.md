# 🩺 MedGPT - Medical Knowledge Assistant

Professional medical RAG (Retrieval-Augmented Generation) assistant with semantic search, AI-powered responses, and PDF document viewer.

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-121212?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)

## ✨ Features

- 🔍 **Semantic Search** - Find relevant medical information using AI-powered embeddings
- 🤖 **Multiple LLM Backends** - Groq (recommended), OpenAI, Ollama, or smart fallback
- 📚 **Source Citations** - Every answer linked to source documents with page numbers
- 📄 **PDF Viewer** - Inline document viewing with highlighted relevant sections
- 📤 **File Upload** - Add new medical documents dynamically
- ☁️ **Cloud Ready** - Optimized for Streamlit Cloud deployment
- ⚡ **Fast** - Groq API provides 800+ tokens/second inference speed

## 🚀 Quick Start

### Local Setup (5 minutes)

```bash
# 1. Clone the repository
git clone https://github.com/your-username/Med_GPT.git
cd Med_GPT

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment
cp .env.template .env
# Edit .env and add your Groq API key

# 4. Prepare documents
mkdir documents
# Add your medical PDFs to documents/
python utils/document_processor.py

# 5. Run the app
streamlit run app.py
```

### Get Free Groq API Key

1. Visit [console.groq.com](https://console.groq.com)
2. Sign up (it's free!)
3. Create an API key
4. Add to `.env`: `GROQ_API_KEY="gsk_your_key"`

## 📁 Project Structure

```
Med_GPT/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── packages.txt               # System packages (for Streamlit Cloud)
├── .env.template              # Environment variables template
├── .streamlit/
│   └── config.toml           # Streamlit configuration
├── documents/                 # Your medical PDFs (local)
├── vectorstore/              # FAISS vector database
│   ├── index.faiss
│   └── index.pkl
├── student_uploads/          # User-uploaded documents
└── utils/
    ├── llm_handler.py        # LLM backend management
    ├── gdrive_loader.py      # HuggingFace vectorstore download
    ├── document_processor.py # PDF processing & indexing
    └── upload_handler.py     # File upload interface

# Documentation
├── README.md                  # This file
├── SETUP_GUIDE.md            # Detailed setup instructions
├── QUICKSTART.md             # 5-minute quick start
└── STREAMLIT_CLOUD_DEPLOYMENT.md  # Cloud deployment guide

# Setup Scripts
├── setup.py                  # Automated setup script
└── check_deployment.py       # Pre-deployment validation
```

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **LLM**: Groq API (Llama 3.3 70B), OpenAI GPT-3.5, Ollama (local)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: FAISS
- **Document Processing**: PyMuPDF, PyPDF2, python-docx
- **Framework**: LangChain

## 📋 Requirements

- Python 3.9 or higher
- 4GB RAM minimum (8GB recommended)
- Internet connection (for API calls)
- Git (for version control)

## 🎯 Usage

### 1. Ask a Question

Type a medical query in the search box:
```
"What are the treatment options for acute myocardial infarction?"
"Explain the mechanism of action of ACE inhibitors"
"What are the diagnostic criteria for diabetes?"
```

### 2. View Sources

Click on any source reference to:
- View the exact excerpt used
- See the PDF page with highlighted text
- Navigate through the document

### 3. Upload Documents

Use the sidebar to:
- Upload new PDFs or DOCX files
- Process and add to knowledge base
- Immediately available for search

## 🌐 Streamlit Cloud Deployment

### Quick Deploy

1. **Push to GitHub**
   ```bash
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository
   - Add API key in Secrets:
     ```toml
     GROQ_API_KEY = "gsk_your_key"
     ```

3. **Done!** Your app is live at `your-app.streamlit.app`

### Pre-Deployment Check

```bash
python check_deployment.py
```

This validates:
- ✅ All required files present
- ✅ Git repository configured
- ✅ Dependencies listed
- ✅ Vectorstore ready
- ✅ Configuration files in place

See [STREAMLIT_CLOUD_DEPLOYMENT.md](STREAMLIT_CLOUD_DEPLOYMENT.md) for detailed instructions.

## 🔧 Configuration

### LLM Backend Priority

1. **Groq API** (Default) - Fast, free tier available
2. **OpenAI API** - Reliable, paid service
3. **Ollama** - Local, private (local deployment only)
4. **Fallback** - Context extraction without LLM

Configure in `utils/llm_handler.py`.

### Adjust Search Parameters

In `app.py`:
```python
TOP_K = 3  # Number of relevant chunks to retrieve
```

In `utils/document_processor.py`:
```python
CHUNK_SIZE = 1000      # Size of text chunks
CHUNK_OVERLAP = 200    # Overlap between chunks
```

## 📊 Performance

- **Groq API**: 800+ tokens/second
- **Local Embeddings**: ~100ms per query
- **Vector Search**: <50ms for 10K documents
- **PDF Rendering**: Instant with caching

## 🐛 Troubleshooting

### "Groq API initialization failed"

**Solution**: 
1. Check your API key in `.env` or Streamlit Secrets
2. Verify key starts with `gsk_`
3. Test at [console.groq.com](https://console.groq.com)

### "No vectorstore found"

**Solution**:
```bash
# Option 1: Build locally
mkdir documents
# Add PDFs to documents/
python utils/document_processor.py

# Option 2: Configure HuggingFace download
# Edit utils/gdrive_loader.py with your repo_id
```

### "PDF preview not available"

**Expected on Streamlit Cloud**. The app automatically uses:
1. PyMuPDF rendering (local)
2. Base64 embedded viewer (cloud)
3. Download link (fallback)

### "Limited Mode" - No AI responses

**Solution**: Configure an API key
- Preferred: Groq API (free, fast)
- Alternative: OpenAI API
- The app works without AI but responses are basic

## 📚 Documentation

- 📖 [SETUP_GUIDE.md](SETUP_GUIDE.md) - Comprehensive setup instructions
- ⚡ [QUICKSTART.md](QUICKSTART.md) - 5-minute quick start
- ☁️ [STREAMLIT_CLOUD_DEPLOYMENT.md](STREAMLIT_CLOUD_DEPLOYMENT.md) - Cloud deployment
- 🔧 Run `python setup.py` for automated setup
- ✅ Run `python check_deployment.py` before deploying

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally and on Streamlit Cloud
5. Submit a pull request

## ⚠️ Important Notes

- **Educational Use Only** - Not for clinical decision-making
- **Verify Information** - Always cross-reference medical information
- **Privacy** - Keep API keys private, never commit to Git
- **Data Security** - Don't upload sensitive patient information

## 📝 License

This project is provided as-is for educational purposes.

## 🙏 Acknowledgments

- **Groq** - For providing fast, free LLM inference
- **Streamlit** - For the amazing framework
- **LangChain** - For RAG tooling
- **HuggingFace** - For embeddings models

## 📧 Support

- 📖 Check documentation first
- 🐛 Report issues on GitHub
- 💬 Ask questions in discussions
- 🌟 Star the repo if you find it useful!

---

**Built with ❤️ for medical education**

**Remember**: This is an educational tool. Always verify medical information with qualified healthcare professionals.
