# DocuChat: AI-Powered PDF Q&A 📄💬

DocuChat is a web application that lets you have a conversation with your documents. Simply upload one or more PDFs, and ask any questions you have about their content. The app uses the power of Large Language Models (LLMs) to provide accurate, context-aware answers.

![DocuChat Demo GIF](https://your-gif-url-here.gif) ## ✨ Features

-   **Interactive Chat Interface:** A user-friendly interface for asking questions.
-   **Multiple Document Support:** Upload and query multiple PDFs at once.
-   **Context-Aware Answers:** The model answers questions based *only* on the information present in the uploaded documents, which prevents it from making things up.
-   **Session Memory:** Remembers previous questions and answers in the current session.

## ⚙️ How It Works (Architecture)

This project uses a **Retrieval-Augmented Generation (RAG)** architecture:

1.  **PDF Processing:** The text from the uploaded PDFs is extracted.
2.  **Text Chunking:** The extracted text is divided into smaller, manageable chunks.
3.  **Vector Embeddings:** Each chunk is converted into a numerical representation (a vector) using OpenAI's embedding models.
4.  **Vector Store:** These vectors are stored in a FAISS vector store, which allows for efficient similarity searching.
5.  **Q&A Process:** When you ask a question, it's also converted into a vector. The system finds the most relevant text chunks from the vector store and provides them to the LLM as context, along with your question, to generate a precise answer.



[Image of a Retrieval-Augmented Generation diagram]


## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

-   Python 3.8+
-   An [OpenAI API Key](https.platform.openai.com/account/api-keys)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/docuchat-ai.git](https://github.com/your-username/docuchat-ai.git)
    cd docuchat-ai
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**
    -   Create a file named `.env` by duplicating `.env.example`.
    -   Add your OpenAI API key to the `.env` file:
    ```
    OPENAI_API_KEY="sk-YourSecretAPIKeyGoesHere"
    ```

### Running the Application

1.  **Launch the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

2.  Open your web browser and navigate to `http://localhost:8501`.

## 🛠️ Built With

-   [Python](https://www.python.org/)
-   [Streamlit](https://streamlit.io/) - Web framework
-   [LangChain](https://www.langchain.com/) - AI orchestration
-   [OpenAI](https://openai.com/) - LLM and embedding models
-   [FAISS](https://github.com/facebookresearch/faiss) - Vector store
-   [PyPDF2](https://pypi.org/project/PyPDF2/) - PDF text extraction

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.