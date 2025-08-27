## AI Teaching Assistant with Retrieval-Augmented Generation (RAG)

This project implements an AI Teaching Assistant using Retrieval-Augmented Generation (RAG) with the LLaMA-2 7B model. It processes course files, creates a vector database, and provides a Streamlit-based interface for student interaction. The repository includes Jupyter Notebooks for implementation and evaluation, and a Streamlit app.

# Workflows

1. Set up the environment and install dependencies
2. Run `SEP_775_AI_TA_Core_Implementation.ipynb` to download the model and set up the RAG pipeline
3. Run `SEP_775_AI_TA_Evaluation_Code.ipynb` to evaluate performance
4. Run `app.py` to launch the Streamlit app
5. Deploy locally and interact with the AI TA

# How to run?

### STEPS:

Clone the repository

```bash
git clone https://github.com/Rutvik46/AI-Teaching-Assistant-Retrieval-Augmented-Generation-
cd AI-Teaching-Assistant-Retrieval-Augmented-Generation-
```

### STEP 01- Create a virtual environment after opening the repository

```bash
python -m venv venv
```

```bash
.\venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On macOS/Linux
```

### STEP 02- Install the requirements

```bash
pip install -r requirements.txt
```

Required packages:
- langchain==0.1.14
- sentence-transformers==2.6.1
- faiss-cpu==1.8.0
- pdfminer.six==20231228
- llama-cpp-python==0.2.58
- tiktoken==0.6.0
- gdown==5.1.0

Alternatively, install manually:

```bash
pip install langchain==0.1.14 sentence-transformers==2.6.1 faiss-cpu==1.8.0 pdfminer.six==20231228 llama-cpp-python==0.2.58 tiktoken==0.6.0 gdown==5.1.0
```

### STEP 03- Run the core implementation

Execute `SEP_775_AI_TA_Core_Implementation.ipynb` to:
- Download the LLaMA-2 7B model (`llama-2-7b-chat.Q4_K_M.gguf`)
- Extract text from course files
- Create a vector database with `faiss-cpu`
- Set up the LLaMA-2 model and question-answering chain

**Note**: If verbose logs (e.g., token generation times) appear, set `verbose=False` in the `LlamaCPP` class.

### STEP 04- Evaluate the AI TA

Run `SEP_775_AI_TA_Evaluation_Code.ipynb` to:
- Load a CSV with question-answer pairs
- Compute metrics (e.g., BLEU score, response time)
- Test hyperparameters (see project report for details)

### STEP 05- Run the Streamlit app

```bash
cd "A:\SEP 775 NLP\SEP 775 Project - AI Teaching Assistant (RAG)\AI Teaching Assistant (Retrieval-Augmented Generation)"
streamlit run app.py
```

Now,
```bash
open up your local host and port (e.g., http://localhost:8501)
```
Mac_AI_Teaching_Assistant.png

**Note**: Ensure `llama-2-7b-chat.Q4_K_M.gguf` is in the project directory and the path in `app.py` is correct.

# Project Structure

- `SEP_775_AI_TA_Core_Implementation.ipynb`: Downloads the LLaMA-2 model, processes course files, and sets up the RAG pipeline
- `SEP_775_AI_TA_Evaluation_Code.ipynb`: Evaluates the AI TA with question-answer pairs and performance metrics
- `app.py`: Streamlit app for user interaction, generating:
  - `user_files/`: Stores user-uploaded files
  - `vectorstore/`: Contains the vector database index
- `.gitignore`: Excludes `venv/`, `llama-2-7b-chat.Q4_K_M.gguf`, and other temporary files
- `requirements.txt`: Lists dependencies for reproducibility

# Troubleshooting

### `llama-cpp-python` installation error

If you see "Failed building wheel for llama-cpp-python":
- Install Visual Studio Build Tools:
  1. Download from [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools)
  2. Select **Desktop development with C++** and **Windows 10 SDK**
- Reinstall:

```bash
pip install llama-cpp-python==0.2.58
```

- See [Stack Overflow](https://stackoverflow.com/questions/77267346/error-while-installing-python-package-llama-cpp-python)

### Model not found

If `llama-2-7b-chat.Q4_K_M.gguf` is not found:
- Ensure it’s in the project directory
- Update the file path in `app.py` or `SEP_775_AI_TA_Core_Implementation.ipynb`

### Dependency issues

For errors with `langchain`, `sentence-transformers`, etc.:
- Verify installation:

```bash
.\venv\Scripts\activate
pip install -r requirements.txt
```

- Check for version mismatches and reinstall if needed.