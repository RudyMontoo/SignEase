# SignEase MVP Quick Start Guide

Get up and running with SignEase in under 5 minutes! 🚀

## 🎯 Try the Live Demo First

**🌐 [SignEase Live Demo](https://signease-mvp.vercel.app)**

1. Click the link above
2. Allow camera permissions
3. Position your hand in view
4. Sign ASL letters A-Z
5. Watch real-time recognition!

## ⚡ 5-Minute Local Setup

### Step 1: Prerequisites Check
```bash
# Check Node.js (need 18+)
node --version

# Check Python (need 3.9+)
python --version

# Check npm
npm --version
```

Don't have these? Quick install:
- **Node.js**: [Download here](https://nodejs.org/)
- **Python**: [Download here](https://python.org/)

### Step 2: Get the Code
```bash
# Clone repository
git clone https://github.com/your-org/signease-mvp.git
cd signease-mvp
```

### Step 3: Start Frontend (Terminal 1)
```bash
cd signease-frontend
npm install
npm run dev
```
✅ Frontend running at: http://localhost:5173

### Step 4: Start Backend (Terminal 2)
```bash
cd backend
pip install -r requirements.txt
python app.py
```
✅ Backend running at: http://localhost:8000

### Step 5: Test It!
1. Open http://localhost:5173
2. Allow camera permissions
3. Sign the letter "A" in ASL
4. See real-time recognition! 🎉

## 🎮 Demo Instructions

### Basic Usage
1. **Position Hand**: Center your hand in the camera view
2. **Good Lighting**: Ensure adequate lighting (not too bright/dark)
3. **Clear Background**: Plain background works best
4. **Steady Hand**: Hold gesture for 1-2 seconds
5. **Watch Magic**: See real-time recognition with AR overlay!

### Try These Gestures
- **A**: Closed fist with thumb on side
- **B**: Open hand, fingers together, thumb across palm
- **C**: Curved hand like holding a cup
- **Hello**: Spell H-E-L-L-O letter by letter
- **Love**: L-O-V-E for a complete word

### Features to Explore
- 🎨 **Themes**: Toggle dark/light mode
- 🔊 **Speech**: Enable text-to-speech
- ✨ **AR Overlay**: Try different display modes
- ⚙️ **Settings**: Adjust confidence thresholds
- 📊 **Performance**: View real-time metrics

## 🛠️ Troubleshooting

### Camera Not Working?
```bash
# Check camera permissions in browser
# Chrome: Settings > Privacy > Camera
# Firefox: Preferences > Privacy > Camera
# Safari: Preferences > Websites > Camera
```

### API Not Responding?
```bash
# Check if backend is running
curl http://localhost:8000/api/health

# Restart backend if needed
cd backend
python app.py
```

### Poor Recognition Accuracy?
- ✅ Ensure good lighting
- ✅ Use plain background
- ✅ Hold gestures steady
- ✅ Check camera focus
- ✅ Try adjusting confidence threshold

### Performance Issues?
```bash
# Check system resources
# Close other browser tabs
# Restart browser
# Check GPU availability
```

## 🎯 Demo Script (2 Minutes)

Perfect for showing SignEase to others:

### Minute 1: Introduction
> "Hi! I'm going to show you SignEase - a real-time ASL recognition system. Watch as I sign the letter 'A'..."

*Sign letter A, show real-time recognition*

> "Notice the instant recognition with 95%+ confidence and the AR overlay showing the letter directly over my hand."

### Minute 2: Advanced Features
> "Now let me spell 'HELLO' to show sentence building..."

*Sign H-E-L-L-O, demonstrate sentence builder*

> "The system builds complete words and can even speak them aloud. Let me show the settings..."

*Quickly show control panel, AR modes, speech settings*

> "SignEase achieves 99.57% accuracy and processes gestures in under 100ms. Questions?"

## 📱 Mobile Usage

### Supported Devices
- ✅ **Desktop**: Full functionality (recommended)
- ✅ **Tablet**: Good performance with touch controls
- ⚠️ **Mobile**: Limited performance, basic functionality

### Mobile Optimization
- Use landscape orientation
- Ensure stable internet connection
- Close other apps for better performance
- Use external lighting if needed

## 🔧 Advanced Configuration

### Environment Variables

**Frontend (.env)**
```env
# API Configuration
VITE_API_BASE_URL=http://localhost:8000
VITE_API_TIMEOUT=10000

# Feature Flags
VITE_ENABLE_PERFORMANCE_MONITORING=true
VITE_ENABLE_DEBUG_MODE=false
VITE_ENABLE_ANALYTICS=false

# UI Configuration
VITE_DEFAULT_THEME=light
VITE_ENABLE_ANIMATIONS=true
VITE_SHOW_FPS_COUNTER=true
```

**Backend (.env)**
```env
# Model Configuration
MODEL_PATH=./models/asl_model_best_20251102_214717.json
CONFIDENCE_THRESHOLD=0.7
MAX_ALTERNATIVES=3

# Performance Configuration
ENABLE_GPU=true
MAX_BATCH_SIZE=10
CACHE_TTL=300
MAX_CONCURRENT_REQUESTS=100

# Logging
LOG_LEVEL=INFO
ENABLE_METRICS=true
```

### Custom Model

Want to use your own trained model?

1. **Train Model**
   ```bash
   cd backend
   python train.py --dataset your_dataset.json --epochs 100
   ```

2. **Update Configuration**
   ```env
   MODEL_PATH=./models/your_custom_model.json
   ```

3. **Restart Backend**
   ```bash
   python app.py
   ```

## 🎨 Customization

### Themes
```typescript
// Custom theme configuration
const customTheme = {
  colors: {
    primary: '#your-color',
    secondary: '#your-color',
    background: '#your-color'
  },
  fonts: {
    primary: 'Your Font Family'
  }
}
```

### UI Components
```typescript
// Custom component styling
<GestureDisplay 
  theme="custom"
  showConfidence={true}
  overlayMode="floating"
  animationSpeed="fast"
/>
```

## 📊 Monitoring

### Performance Dashboard
Access real-time performance metrics:
- Frontend: Click the ⚙️ settings icon → Performance tab
- Backend: http://localhost:8000/metrics

### Key Metrics to Watch
- **FPS**: Should stay above 25 FPS
- **Latency**: Should be under 100ms
- **Accuracy**: Should be above 90%
- **Memory**: Should be under 2GB

## 🆘 Getting Help

### Quick Help
- **Camera Issues**: Check browser permissions
- **Performance Issues**: Close other tabs, restart browser
- **Recognition Issues**: Improve lighting, steady hand
- **API Issues**: Check backend is running

### Detailed Help
- **Documentation**: README.md, API_DOCUMENTATION.md
- **GitHub Issues**: Report bugs and request features
- **Community**: Join our Discord for real-time help
- **Email**: support@signease.dev

## 🎉 Success!

If you can see real-time ASL recognition working, you're all set! 

### What's Next?
- Explore all 26 ASL letters
- Try building complete sentences
- Experiment with different settings
- Share with friends and family
- Consider contributing to the project

---

**Welcome to SignEase! Start breaking down communication barriers today.** 🤟

*Questions? We're here to help at support@signease.dev*