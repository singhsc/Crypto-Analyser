# 🔐 Cryptocurrency Wallet Analyzer - Real Data Edition

**Complete ML system with 500K+ real cryptocurrency records from Kaggle, GitHub, Hugging Face, and GitLab**

## 📊 What's Included

### Datasets (500K+ Real Records)

1. **Ethereum Fraud Detection Dataset**
   - 9,841 Ethereum accounts
   - Fraud labels and wallet metrics
   - Source: Kaggle

2. **CryptoXChain-500K Dataset**
   - 500,000 real blockchain transactions
   - Multi-network data (Ethereum, Bitcoin, etc.)
   - Source: Hugging Face

3. **Elliptic++ Dataset**
   - 203,769 Bitcoin transactions
   - 822,482 wallet addresses
   - Transaction graph network
   - Fraud labels
   - Source: GitHub

4. **Pump-and-Dump Events Dataset**
   - 10,687 confirmed pump events
   - Price and volume data
   - Source: GitLab

### ML Models

- **KMeans Clustering**: 5 wallet behavior clusters
- **IsolationForest**: Anomaly detection with risk scoring
- **Manipulation Detectors**: 3 specialized detectors
- **Model Evaluation**: Silhouette, Davies-Bouldin, F1, ROC-AUC metrics

### Dashboard (5 Pages)

1. **Overview**: Dataset statistics, risk distribution
2. **Wallets**: Search and analyze individual wallets
3. **Alerts**: Market manipulation detection results
4. **Models**: Training metrics and performance
5. **Analytics**: Real data visualizations and patterns

---

## 🚀 Quick Start

### Option 1: Automatic Setup (Recommended)

**Mac/Linux:**
```bash
bash run_real_data.sh
# Choose option 1 to launch dashboard
```

**Windows:**
```batch
run_real_data.bat
REM Choose option 1 to launch dashboard
```

### Option 2: Manual Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements_complete.txt

# Download datasets (see instructions below)
python DATASET_SETUP_GUIDE.py

# Train models on real data
python model_training.py

# Launch dashboard
streamlit run crypto_analyzer_real_data.py
```

---

## 📥 Download Datasets

### Quick Links

1. **Ethereum Fraud Detection**
   - URL: https://www.kaggle.com/datasets/vagifa/ethereum-fraudetection-dataset
   - Command: `kaggle datasets download -d vagifa/ethereum-fraudetection-dataset`

2. **CryptoXChain-500K**
   - URL: https://huggingface.co/datasets/Omarrran/CryptoXChain_500K_Multi_Network_Blockchain_Transaction_Dataset
   - Command: Uses Hugging Face Hub (automatic)

3. **Elliptic++**
   - URL: https://github.com/git-disl/EllipticPlusPlus
   - Command: `git clone https://github.com/git-disl/EllipticPlusPlus.git`

4. **Pump-and-Dump**
   - URL: https://gitlab.com/bristol-university-work/crypto-pump-and-dump
   - Command: `git clone https://gitlab.com/bristol-university-work/crypto-pump-and-dump.git`

### Detailed Instructions

Run the setup guide for step-by-step instructions:
```bash
python DATASET_SETUP_GUIDE.py
```

This will:
- Check dependencies (Kaggle API, Git LFS, etc.)
- Provide download links
- Explain file placement
- Verify dataset integrity

---

## 🎯 System Workflow

```
┌─────────────────────────────────────┐
│  Download Real Datasets             │
│  (500K+ records)                    │
└────────────┬────────────────────────┘
             ↓
┌─────────────────────────────────────┐
│  Load & Unify Data                  │
│  (common schema)                    │
└────────────┬────────────────────────┘
             ↓
┌─────────────────────────────────────┐
│  Train ML Models                    │
│  - KMeans clustering                │
│  - IsolationForest anomaly          │
│  - Risk scoring                     │
└────────────┬────────────────────────┘
             ↓
┌─────────────────────────────────────┐
│  Detect Manipulations               │
│  - Pump & Dump                      │
│  - Wash Trading                     │
│  - Whale Manipulation               │
└────────────┬────────────────────────┘
             ↓
┌─────────────────────────────────────┐
│  Interactive Dashboard              │
│  (5 pages, real data)               │
└─────────────────────────────────────┘
```

---

## 📊 Models & Metrics

### Clustering (KMeans)

Metrics calculated:
- **Silhouette Score**: -1 to 1 (higher is better)
- **Davies-Bouldin Score**: 0 to ∞ (lower is better)
- **Calinski-Harabasz Score**: 0 to ∞ (higher is better)

Interpretation:
- Silhouette > 0.5: Good cluster separation
- Davies-Bouldin < 1: Compact, well-separated clusters
- Calinski-Harabasz > 300: Strong cluster definition

### Anomaly Detection (IsolationForest)

Metrics calculated:
- **Precision**: Ratio of correct positive predictions
- **Recall**: Ratio of detected anomalies
- **F1 Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under ROC curve

Interpretation:
- F1 > 0.7: Good detection performance
- ROC AUC > 0.8: Excellent discrimination
- Adjust contamination for precision/recall tradeoff

---

## 🔧 Configuration

Edit `crypto_analyzer_real_data.py`:

```python
class Config:
    # Use real data (True) or mock data (False)
    USE_REAL_DATA = True
    
    # Maximum samples to process (for speed)
    MAX_SAMPLES = 5000
    
    # ML Parameters
    N_CLUSTERS = 5
    ANOMALY_CONTAMINATION = 0.1
    
    # Thresholds
    LOW_RISK_THRESHOLD = 30
    MEDIUM_RISK_THRESHOLD = 70
```

---

## 📁 Directory Structure

```
crypto-wallet-analyzer/
├── crypto_analyzer.py                 # Mock data version
├── crypto_analyzer_real_data.py       # Real data version
├── dataset_loader.py                  # Dataset loading & unification
├── model_training.py                  # ML model training
├── DATASET_SETUP_GUIDE.py            # Dataset download guide
│
├── requirements_minimal.txt           # Basic dependencies
├── requirements_complete.txt          # All dependencies
│
├── run_real_data.sh                  # Mac/Linux launcher
├── run_real_data.bat                 # Windows launcher
├── run.sh                            # Original Mac/Linux
├── run.bat                           # Original Windows
│
├── data/
│   ├── raw/                          # Raw data
│   ├── processed/                    # Processed data
│   └── real_datasets/                # Downloaded datasets
│       ├── ethereum_fraud_detection.csv
│       ├── cryptoxchain_500k.parquet
│       ├── elliptic_txs_edgelist.csv
│       ├── elliptic_txs_features.csv
│       ├── elliptic_txs_classes.csv
│       └── pump_and_dump_events.csv
│
├── models/                           # Trained ML models
│   ├── kmeans_model.pkl
│   ├── isolation_forest_model.pkl
│   ├── training_metrics.json
│   └── evaluation_report.txt
│
└── logs/                             # Application logs
```

---

## ⚙️ Installation Steps

### Step 1: Clone/Download Project
```bash
# Download the project files
# Extract to your preferred location
cd crypto-wallet-analyzer
```

### Step 2: Install Python Dependencies
```bash
pip install -r requirements_complete.txt
```

### Step 3: Install Tools (Optional)
```bash
# For Kaggle API
pip install kaggle
# Set up credentials: https://www.kaggle.com/settings/account

# For Git LFS (large files)
# macOS: brew install git-lfs
# Linux: sudo apt-get install git-lfs
# Windows: Download from https://git-lfs.com/
```

### Step 4: Download Datasets
```bash
python DATASET_SETUP_GUIDE.py
```

Follow the instructions to download each dataset.

### Step 5: Train Models
```bash
python model_training.py
```

This trains all ML models on your real data and generates evaluation metrics.

### Step 6: Launch Dashboard
```bash
streamlit run crypto_analyzer_real_data.py
```

Dashboard opens at: `http://localhost:8501`

---

## 📈 Dashboard Pages

### 1. Overview
- Total wallets and transactions
- Data source breakdown
- Risk distribution charts
- Dataset statistics

### 2. Wallets
- Search any wallet
- View detailed metrics
- Transaction history
- Cluster assignment
- Risk scoring

### 3. Alerts
- All detected manipulations
- Alert type breakdown (Pump & Dump, Wash Trading, Whale)
- Risk scores
- Descriptions and timestamps

### 4. Models
- Training metrics
- Silhouette scores
- Anomaly detection statistics
- Model information

### 5. Analytics
- Volume distribution
- Transaction patterns
- Risk analysis
- Cluster analysis
- Real data visualizations

---

## 🧪 Testing & Validation

### Validate Dataset Setup
```bash
python -c "
import pandas as pd
import os

datasets = {
    'ethereum': 'data/real_datasets/ethereum_fraud_detection.csv',
    'cryptoxchain': 'data/real_datasets/cryptoxchain_500k.parquet',
    'elliptic_edges': 'data/real_datasets/elliptic_txs_edgelist.csv',
    'pump_dump': 'data/real_datasets/pump_and_dump_events.csv'
}

for name, path in datasets.items():
    if os.path.exists(path):
        print(f'✓ {name}')
    else:
        print(f'✗ {name} NOT FOUND')
"
```

### Test Model Training
```bash
python model_training.py
```

This will train models and output evaluation metrics.

### Test Dashboard
```bash
streamlit run crypto_analyzer_real_data.py
```

Check all 5 pages load correctly with real data.

---

## 🐛 Troubleshooting

### No Datasets Found
**Solution:**
```bash
python DATASET_SETUP_GUIDE.py
# Follow the setup instructions
```

### Memory Error (Too Many Records)
**Solution:**
Edit `crypto_analyzer_real_data.py`:
```python
Config.MAX_SAMPLES = 1000  # Reduce from 5000
```

### Slow Training
**Solution:**
```python
# Use smaller sample
Config.MAX_SAMPLES = 500

# Or reduce clusters
Config.N_CLUSTERS = 3
```

### Kaggle API Not Found
**Solution:**
```bash
pip install kaggle
# Set up credentials at https://www.kaggle.com/settings/account
```

### Git LFS Not Installed
**Solution:**
```bash
# macOS
brew install git-lfs

# Ubuntu
sudo apt-get install git-lfs

# Windows
Download from https://git-lfs.com/
```

---

## 📊 Expected Results

With all datasets loaded:

| Metric | Value |
|--------|-------|
| Total Records | 500,000+ |
| Wallets Analyzed | 10,000+ |
| Transactions Processed | 500,000+ |
| Clusters Identified | 5 |
| Anomalies Detected | 5-10% |
| Manipulation Alerts | 50-200 |
| Model Training Time | 2-5 minutes |
| Dashboard Load Time | <5 seconds |

---

## 🎓 Learning Resources

### Dataset Papers
- Ethereum Fraud: https://arxiv.org/abs/2101.05511
- Elliptic++: https://arxiv.org/abs/2201.07220

### ML Concepts
- KMeans Clustering: https://scikit-learn.org/stable/modules/clustering.html#k-means
- Isolation Forest: https://scikit-learn.org/stable/modules/ensemble.html#isolation-forest
- Anomaly Detection: https://en.wikipedia.org/wiki/Anomaly_detection

### Tools
- Streamlit: https://docs.streamlit.io/
- Plotly: https://plotly.com/python/
- Scikit-learn: https://scikit-learn.org/

---

## 💡 Tips & Tricks

### 1. Fast Startup
```python
# Use smaller sample for quick testing
Config.MAX_SAMPLES = 1000
Config.N_CLUSTERS = 3
```

### 2. High Precision Alerts
```python
# Increase anomaly threshold
Config.LOW_RISK_THRESHOLD = 50
Config.MEDIUM_RISK_THRESHOLD = 80
```

### 3. High Recall Alerts
```python
# Decrease anomaly threshold
Config.LOW_RISK_THRESHOLD = 20
Config.MEDIUM_RISK_THRESHOLD = 50
```

### 4. Custom Features
Edit `dataset_loader.py` to add custom features to the unified schema.

### 5. Export Results
```python
import pandas as pd
results_df = pd.read_csv('models/results.csv')
results_df.to_excel('analysis.xlsx')
```

---

## 🚀 Next Steps

1. **Download Datasets** (if not already done)
   ```bash
   python DATASET_SETUP_GUIDE.py
   ```

2. **Train Models**
   ```bash
   python model_training.py
   ```

3. **Launch Dashboard**
   ```bash
   streamlit run crypto_analyzer_real_data.py
   ```

4. **Explore Results**
   - Click through all 5 pages
   - Analyze wallets
   - Review alerts
   - Study metrics

5. **Customize**
   - Modify thresholds
   - Add features
   - Connect to production APIs

---

## 📝 License

MIT License - Free for personal and commercial use

---

## 🤝 Contributing

To improve this project:
1. Add new datasets
2. Implement new detectors
3. Improve visualizations
4. Optimize performance
5. Add features

---

## 📞 Support

For help:
1. Check DATASET_SETUP_GUIDE.py
2. Review error messages
3. Check model_training.py output
4. Verify dataset files
5. Test with mock data first

---

## ✨ Features Summary

✅ **500K+ Real Records** from 4 major datasets  
✅ **Unified Schema** combining multiple data sources  
✅ **ML Models** with evaluation metrics  
✅ **Interactive Dashboard** with real data visualizations  
✅ **Model Training** scripts with complete pipeline  
✅ **Automatic Dataset Loading** and preprocessing  
✅ **Risk Scoring** with multiple detection methods  
✅ **Export Capabilities** for further analysis  
✅ **Evaluation Metrics** (Silhouette, F1, ROC-AUC)  
✅ **Production Ready** code with logging  

---

**Ready to analyze 500K+ real cryptocurrency records?**

```bash
bash run_real_data.sh  # Mac/Linux
# or
run_real_data.bat     # Windows
```

Dashboard opens at: `http://localhost:8501` 🚀
