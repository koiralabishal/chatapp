# AgroMart AI - RAG PDF Chat Application

A premium, production-ready Chat Application that uses **Retrieval-Augmented Generation (RAG)** to answer questions based on your PDF documents. Built with **FastAPI**, **Google Gemini AI**, and a modern **Vanilla JS/CSS** frontend.

## 🚀 Features

- **RAG Integration**: Automatically reads and processes PDF documents from the `/data` folder.
- **Google Gemini AI**: Powered by `gemini-3-flash-preview` for fast and accurate responses.
- **Premium UI**: Modern dark-themed interface with glassmorphism, bouncing dot typing indicators, and smooth animations.
- **Streaming Responses**: Real-time AI response delivery for a conversational feel.
- **Vercel Ready**: Fully configured for one-click deployment to Vercel as a Serverless function.

## 🛠️ Technology Stack

- **Backend**: Python, FastAPI, Uvicorn
- **AI/LLM**: Google Gen AI (Gemini 1.5 & Embedding models)
- **Frontend**: HTML5, Vanilla CSS, Vanilla JavaScript
- **Deployment**: Vercel (Static + Serverless)

## 📦 Installation

1. **Clone the repository**:

   ```bash
   git clone git@github.com:koiralabishal/chatapp.git
   cd chatapp
   ```

2. **Set up Virtual Environment**:

   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**:
   Create a `.env` file in the root directory and add your Gemini API Key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

## 🏃 Local and VercelDevelopment

1. **Start the Backend**:

   ```bash
   python api/index.py
   ```

   _The server will run at `http://127.0.0.1:8001`._

2. **Open the Frontend**:
   Open `public/index.html` in your browser (or use VS Code Live Server).

## ☁️ Deployment to Vercel

1. **Connect to Vercel**: Import your repository to the Vercel dashboard.
2. **Configure Environment Variables**: Add `GEMINI_API_KEY` in Vercel Project Settings.
3. **Deploy**: Vercel will automatically use `vercel.json` to route `/chat` to the serverless function.

## 📂 Project Structure

- `/api`: Vercel serverless function entry point (`index.py`).
- `/public`: Frontend assets (`index.html`).
- `/data`: Place your PDF guides here.
- `api/index.py`: Main script for local and vercel development.
- `vercel.json`: Deployment configuration.

## 📄 License

This project is open-source and available under the MIT License.
