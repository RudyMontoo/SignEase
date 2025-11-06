# SignEase MVP Deployment Guide

This guide provides comprehensive instructions for deploying SignEase MVP to production environments.

## 🚀 Quick Deploy to Vercel (Recommended)

### Prerequisites
- GitHub account with repository access
- Vercel account (free tier available)
- Domain name (optional)

### One-Click Deploy
[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/your-org/signease-mvp)

### Manual Vercel Deployment

1. **Install Vercel CLI**
   ```bash
   npm i -g vercel
   ```

2. **Login to Vercel**
   ```bash
   vercel login
   ```

3. **Deploy Frontend**
   ```bash
   cd signease-frontend
   vercel --prod
   ```

4. **Configure Environment Variables**
   ```bash
   vercel env add VITE_API_BASE_URL
   vercel env add VITE_APP_NAME
   vercel env add VITE_APP_VERSION
   ```

5. **Deploy Backend (Vercel Functions)**
   ```bash
   cd ../backend
   vercel --prod
   ```

### Environment Configuration

**Frontend Environment Variables**
```env
VITE_API_BASE_URL=https://your-api-domain.vercel.app
VITE_APP_NAME=SignEase MVP
VITE_APP_VERSION=1.0.0
VITE_ENABLE_PERFORMANCE_MONITORING=true
VITE_ENABLE_ANALYTICS=true
```

**Backend Environment Variables**
```env
MODEL_PATH=./models/asl_model_best_20251102_214717.json
ENABLE_GPU=false
LOG_LEVEL=INFO
CORS_ORIGINS=https://your-frontend-domain.vercel.app
ENVIRONMENT=production
```

## 🐳 Docker Deployment

### Docker Compose (Full Stack)

1. **Create docker-compose.yml**
   ```yaml
   version: '3.8'
   
   services:
     frontend:
       build:
         context: ./signease-frontend
         dockerfile: Dockerfile
       ports:
         - "3000:3000"
       environment:
         - VITE_API_BASE_URL=http://backend:8000
       depends_on:
         - backend
   
     backend:
       build:
         context: ./backend
         dockerfile: Dockerfile
       ports:
         - "8000:8000"
       environment:
         - MODEL_PATH=/app/models/asl_model_best_20251102_214717.json
         - ENABLE_GPU=false
         - CORS_ORIGINS=http://localhost:3000
       volumes:
         - ./backend/models:/app/models
   ```

2. **Frontend Dockerfile**
   ```dockerfile
   # signease-frontend/Dockerfile
   FROM node:18-alpine AS builder
   
   WORKDIR /app
   COPY package*.json ./
   RUN npm ci --only=production
   
   COPY . .
   RUN npm run build
   
   FROM nginx:alpine
   COPY --from=builder /app/dist /usr/share/nginx/html
   COPY nginx.conf /etc/nginx/nginx.conf
   
   EXPOSE 3000
   CMD ["nginx", "-g", "daemon off;"]
   ```

3. **Backend Dockerfile**
   ```dockerfile
   # backend/Dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   COPY . .
   
   EXPOSE 8000
   CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

4. **Deploy with Docker Compose**
   ```bash
   docker-compose up -d --build
   ```

## ☁️ Cloud Platform Deployments

### AWS Deployment

#### Frontend (S3 + CloudFront)
1. **Build for Production**
   ```bash
   cd signease-frontend
   npm run build
   ```

2. **Create S3 Bucket**
   ```bash
   aws s3 mb s3://signease-frontend-prod
   aws s3 website s3://signease-frontend-prod --index-document index.html
   ```

3. **Upload Files**
   ```bash
   aws s3 sync dist/ s3://signease-frontend-prod --delete
   ```

4. **Setup CloudFront Distribution**
   ```json
   {
     "Origins": [{
       "DomainName": "signease-frontend-prod.s3.amazonaws.com",
       "Id": "S3-signease-frontend-prod",
       "S3OriginConfig": {
         "OriginAccessIdentity": ""
       }
     }],
     "DefaultCacheBehavior": {
       "TargetOriginId": "S3-signease-frontend-prod",
       "ViewerProtocolPolicy": "redirect-to-https"
     }
   }
   ```

#### Backend (ECS Fargate)
1. **Create ECR Repository**
   ```bash
   aws ecr create-repository --repository-name signease-backend
   ```

2. **Build and Push Docker Image**
   ```bash
   docker build -t signease-backend ./backend
   docker tag signease-backend:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/signease-backend:latest
   docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/signease-backend:latest
   ```

3. **Create ECS Task Definition**
   ```json
   {
     "family": "signease-backend",
     "networkMode": "awsvpc",
     "requiresCompatibilities": ["FARGATE"],
     "cpu": "256",
     "memory": "512",
     "containerDefinitions": [{
       "name": "signease-backend",
       "image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/signease-backend:latest",
       "portMappings": [{
         "containerPort": 8000,
         "protocol": "tcp"
       }],
       "environment": [
         {"name": "ENABLE_GPU", "value": "false"},
         {"name": "LOG_LEVEL", "value": "INFO"}
       ]
     }]
   }
   ```

### Google Cloud Platform

#### Frontend (Firebase Hosting)
1. **Install Firebase CLI**
   ```bash
   npm install -g firebase-tools
   firebase login
   ```

2. **Initialize Firebase**
   ```bash
   cd signease-frontend
   firebase init hosting
   ```

3. **Configure firebase.json**
   ```json
   {
     "hosting": {
       "public": "dist",
       "ignore": ["firebase.json", "**/.*", "**/node_modules/**"],
       "rewrites": [{
         "source": "**",
         "destination": "/index.html"
       }],
       "headers": [{
         "source": "**/*.@(js|css)",
         "headers": [{
           "key": "Cache-Control",
           "value": "max-age=31536000"
         }]
       }]
     }
   }
   ```

4. **Deploy**
   ```bash
   npm run build
   firebase deploy
   ```

#### Backend (Cloud Run)
1. **Create Dockerfile** (see Docker section above)

2. **Build and Deploy**
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT-ID/signease-backend ./backend
   gcloud run deploy signease-backend \
     --image gcr.io/PROJECT-ID/signease-backend \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

### Microsoft Azure

#### Frontend (Static Web Apps)
1. **Create Static Web App**
   ```bash
   az staticwebapp create \
     --name signease-frontend \
     --resource-group myResourceGroup \
     --source https://github.com/your-org/signease-mvp \
     --location "Central US" \
     --branch main \
     --app-location "/signease-frontend" \
     --output-location "dist"
   ```

#### Backend (Container Instances)
1. **Create Container Registry**
   ```bash
   az acr create --resource-group myResourceGroup --name signeaseregistry --sku Basic
   ```

2. **Build and Push**
   ```bash
   az acr build --registry signeaseregistry --image signease-backend ./backend
   ```

3. **Deploy Container**
   ```bash
   az container create \
     --resource-group myResourceGroup \
     --name signease-backend \
     --image signeaseregistry.azurecr.io/signease-backend:latest \
     --cpu 1 \
     --memory 1 \
     --ports 8000
   ```

## 🔧 Production Configuration

### Security Headers
```nginx
# nginx.conf
server {
    listen 80;
    server_name your-domain.com;
    
    # Security headers
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; connect-src 'self' https:; media-src 'self'; font-src 'self';" always;
    
    # Camera and microphone permissions
    add_header Permissions-Policy "camera=*, microphone=*, geolocation=()" always;
    
    location / {
        root /usr/share/nginx/html;
        try_files $uri $uri/ /index.html;
    }
    
    # API proxy
    location /api/ {
        proxy_pass http://backend:8000/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### SSL/TLS Configuration
```bash
# Let's Encrypt with Certbot
certbot --nginx -d your-domain.com
```

### Performance Optimization
```nginx
# Gzip compression
gzip on;
gzip_vary on;
gzip_min_length 1024;
gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json;

# Browser caching
location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
    expires 1y;
    add_header Cache-Control "public, immutable";
}
```

## 📊 Monitoring and Analytics

### Application Monitoring
```javascript
// Frontend monitoring
import { initializeApp } from 'firebase/app'
import { getAnalytics } from 'firebase/analytics'

const firebaseConfig = {
  // Your config
}

const app = initializeApp(firebaseConfig)
const analytics = getAnalytics(app)
```

### Error Tracking
```javascript
// Sentry integration
import * as Sentry from "@sentry/react"

Sentry.init({
  dsn: "YOUR_SENTRY_DSN",
  environment: "production",
  tracesSampleRate: 1.0,
})
```

### Performance Monitoring
```python
# Backend monitoring
from prometheus_client import Counter, Histogram, generate_latest

REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('request_duration_seconds', 'Request latency')

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    REQUEST_LATENCY.observe(process_time)
    return response
```

## 🔄 CI/CD Pipeline

### GitHub Actions
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - run: npm ci
      - run: npm run test
      - run: npm run build

  deploy-frontend:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - run: npm ci
      - run: npm run build
      - uses: amondnet/vercel-action@v20
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.ORG_ID }}
          vercel-project-id: ${{ secrets.PROJECT_ID }}
          vercel-args: '--prod'

  deploy-backend:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: python -m pytest
      - name: Deploy to Vercel
        uses: amondnet/vercel-action@v20
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.ORG_ID }}
          vercel-project-id: ${{ secrets.BACKEND_PROJECT_ID }}
          vercel-args: '--prod'
          working-directory: ./backend
```

## 🚨 Troubleshooting

### Common Issues

#### Camera Permissions
```javascript
// Check camera permissions
navigator.permissions.query({name: 'camera'}).then(function(result) {
  console.log(result.state); // granted, denied, or prompt
});
```

#### CORS Issues
```python
# Backend CORS configuration
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

#### SSL Certificate Issues
```bash
# Check SSL certificate
openssl s_client -connect your-domain.com:443 -servername your-domain.com
```

### Performance Issues
```bash
# Check resource usage
docker stats
htop
nvidia-smi  # For GPU usage
```

### Debugging
```bash
# View application logs
docker logs container-name
kubectl logs pod-name
vercel logs
```

## 📋 Deployment Checklist

### Pre-Deployment
- [ ] All tests passing
- [ ] Code reviewed and approved
- [ ] Environment variables configured
- [ ] SSL certificates ready
- [ ] Domain DNS configured
- [ ] Monitoring setup complete

### Post-Deployment
- [ ] Application accessible via HTTPS
- [ ] Camera permissions working
- [ ] API endpoints responding
- [ ] Performance metrics normal
- [ ] Error tracking active
- [ ] Backup procedures tested

### Security Checklist
- [ ] HTTPS enforced
- [ ] Security headers configured
- [ ] Input validation implemented
- [ ] Rate limiting active
- [ ] Error messages sanitized
- [ ] Dependencies updated

## 📞 Support

### Deployment Issues
- **Documentation**: Check this guide and README
- **GitHub Issues**: Create issue with deployment details
- **Discord**: Join our community for real-time help
- **Email**: deployment@signease.dev

### Emergency Contacts
- **Critical Issues**: emergency@signease.dev
- **Security Issues**: security@signease.dev
- **Infrastructure**: infrastructure@signease.dev

---

**Deployment made simple for SignEase MVP** 🚀

*Breaking down barriers, one deployment at a time*