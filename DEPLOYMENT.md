# Backend Deployment Guide

## 🚀 **Vercel Deployment from Backend Directory**

Your Vercel project root is now the `backend` directory. Here's how to deploy:

### **📁 Current Structure:**
```
backend/
├── api/
│   ├── index.py          # FastAPI app (Vercel entry point)
│   └── requirements.txt  # Python dependencies
├── api.py                # Original FastAPI app
├── main_local.py         # Local development server
├── vercel.json           # Vercel configuration
├── .vercelignore         # Files to ignore
└── requirements.txt      # Local dependencies
```

### **🚀 Deployment Steps:**

1. **Navigate to Backend Directory:**
   ```bash
   cd backend
   ```

2. **Deploy to Vercel:**
   ```bash
   vercel --prod
   ```

3. **Set Environment Variables in Vercel Dashboard:**
   ```
   OPENAI_API_KEY=sk-your-openai-key-here
   PINECONE_API_KEY=your-pinecone-key
   PINECONE_HOST=https://your-index.svc.pinecone.io
   PINECONE_ENVIRONMENT=us-east-1
   ```

### **🎯 API Endpoints:**
- Health: `https://your-app.vercel.app/api/health`
- Search: `https://your-app.vercel.app/api/search`
- Docs: `https://your-app.vercel.app/api/docs`

### **✅ What's Fixed:**
- ✅ All files in correct backend directory
- ✅ Vercel configuration points to `api/index.py`
- ✅ Dependencies properly configured
- ✅ Error handling for Pinecone initialization

### **🔧 Local Development:**
```bash
# Run local server
python main_local.py
# Server runs on http://localhost:8000
```

Your backend is now ready for Vercel deployment! 🎉

