step-by-step procedure for building the LLM chat application with Hugging Face's Text Generation Inference (TGI) and ElasticSearch for logging. I'll break this down into detailed stages to guide you through the entire development process.

Step-by-Step Procedure for Building the LLM Chat Application

### 1. Project Setup and Initial Planning

#### 1.1 Prerequisites
Before starting, ensure you have the following installed:
- Python 3.9+
- Node.js 16+
- Docker
- pip and npm package managers

#### 1.2 Project Structure Creation
Create a project directory with the following structure:
```
llm-chat-app/
│
├── frontend/           # React application
├── backend/            # FastAPI backend
├── docker-compose.yml  # Container orchestration
└── README.md           # Project documentation
```

### 2. Backend Development (Python with FastAPI)

#### 2.1 Backend Environment Setup
1. Navigate to the backend directory
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

#### 2.2 Install Backend Dependencies
Create a `requirements.txt` with the following dependencies:
```
fastapi==0.104.1
uvicorn==0.24.0
text-generation==0.6.1
elasticsearch==8.11.1
pydantic==2.5.2
python-dotenv==1.0.0
```

Install dependencies:
```bash
pip install -r requirements.txt
```

#### 2.3 Configure ElasticSearch Logging
Create an Elasticsearch client configuration:

```python
from elasticsearch import Elasticsearch
from datetime import datetime
import uuid

class ElasticSearchLogger:
    def __init__(self, host='localhost', port=9200):
        self.es_client = Elasticsearch([{'host': host, 'port': port}])
        self.index_name = 'chat_conversations'
        
        # Create index if not exists
        if not self.es_client.indices.exists(index=self.index_name):
            self.es_client.indices.create(index=self.index_name, 
                body={
                    "mappings": {
                        "properties": {
                            "timestamp": {"type": "date"},
                            "user_message": {"type": "text"},
                            "model_response": {"type": "text"},
                            "model_name": {"type": "keyword"}
                        }
                    }
                })

    def log_conversation(self, user_message, model_response, model_name):
        doc = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow(),
            "user_message": user_message,
            "model_response": model_response,
            "model_name": model_name
        }
        
        self.es_client.index(index=self.index_name, document=doc)

```

#### 2.4 Implement TGI Inference Service
Create a service to interact with Hugging Face's Text Generation Inference:

```python
from text_generation import Client
from typing import Dict, Any

class TGIInferenceService:
    def __init__(self, model_url="http://localhost:8080"):
        self.client = Client(model_url)
    
    def generate_response(
        self, 
        prompt: str, 
        max_tokens: int = 150, 
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        try:
            response = self.client.generate(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature
            )
            return {
                "response": response.generated_text,
                "success": True
            }
        except Exception as e:
            return {
                "response": f"Error generating response: {str(e)}",
                "success": False
            }

```

#### 2.5 FastAPI Backend Implementation
Create the main FastAPI application:

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from .tgi_service import TGIInferenceService
from .elasticsearch_logger import ElasticSearchLogger

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
tgi_service = TGIInferenceService()
es_logger = ElasticSearchLogger()

class ChatRequest(BaseModel):
    prompt: str
    model: str  # Which model to use

class ChatResponse(BaseModel):
    response: str
    model: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    # Generate response
    generation_result = tgi_service.generate_response(request.prompt)
    
    if not generation_result['success']:
        raise HTTPException(status_code=500, detail=generation_result['response'])
    
    # Log conversation
    es_logger.log_conversation(
        user_message=request.prompt,
        model_response=generation_result['response'],
        model_name=request.model
    )
    
    return ChatResponse(
        response=generation_result['response'],
        model=request.model
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

```

### 3. Frontend Development (React)

#### 3.1 Create React Application
```bash
npx create-react-app frontend
cd frontend
npm install axios tailwindcss lucide-react
```

#### 3.2 Frontend Chat Component
Create a comprehensive chat interface:

```tsx
import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Send, RefreshCw } from 'lucide-react';

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [selectedModel, setSelectedModel] = useState('mistral');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const MODELS = {
    mistral: 'Mistral 7B',
    llama: 'Llama 2'
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return;

    const userMessage = {
      text: inputMessage,
      sender: 'user',
      model: selectedModel
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await axios.post('http://localhost:8000/chat', {
        prompt: inputMessage,
        model: selectedModel
      });

      const aiMessage = {
        text: response.data.response,
        sender: 'ai',
        model: response.data.model
      };

      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage = {
        text: 'Error communicating with the model',
        sender: 'system'
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      <div className="flex justify-between p-4 bg-blue-600 text-white">
        <h1 className="text-2xl">Multi-Model LLM Chat</h1>
        <select 
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          className="bg-blue-700 p-2 rounded"
        >
          {Object.entries(MODELS).map(([key, label]) => (
            <option key={key} value={key}>{label}</option>
          ))}
        </select>
      </div>

      <div className="flex-grow overflow-y-auto p-4 space-y-4">
        {messages.map((msg, index) => (
          <div 
            key={index} 
            className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div 
              className={`
                max-w-[70%] p-3 rounded-lg 
                ${msg.sender === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-200'}
              `}
            >
              {msg.text}
              {msg.model && <span className="text-xs block mt-1 opacity-70">
                Model: {MODELS[msg.model] || msg.model}
              </span>}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-gray-200 p-3 rounded-lg flex items-center">
              <RefreshCw className="animate-spin mr-2" />
              Generating response...
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="p-4 bg-white border-t flex">
        <input 
          type="text" 
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
          placeholder="Type your message..."
          className="flex-grow p-2 border rounded-l-lg"
        />
        <button 
          onClick={handleSendMessage}
          disabled={isLoading}
          className="bg-blue-500 text-white p-2 rounded-r-lg hover:bg-blue-600"
        >
          <Send />
        </button>
      </div>
    </div>
  );
};

export default ChatInterface;

```

### 4. Docker Containerization

#### 4.1 Create Docker Compose File
```yaml
version: '3.8'

services:
  frontend:
    build: 
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    depends_on:
      - backend

  backend:
    build: 
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    depends_on:
      - elasticsearch
      - tgi-service

  tgi-service:
    image: ghcr.io/huggingface/text-generation-inference:latest
    ports:
      - "8080:80"
    volumes:
      - model-data:/data
    environment:
      - MODEL_ID=mistralai/Mistral-7B-Instruct-v0.2

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.1
    environment:
      - discovery.type=single-node
    ports:
      - "9200:9200"
    volumes:
      - es-data:/usr/share/elasticsearch/data

volumes:
  model-data:
  es-data:

```

### 5. Deployment and Setup

#### 5.1 Final Setup Steps
1. Create Dockerfiles for frontend and backend
2. Configure environment variables
3. Set up model downloads for TGI
4. Configure Elasticsearch security (for production)

