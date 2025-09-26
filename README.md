# Auto Major Series: Advanced Machine Learning for Automotive Analytics

> **A Comprehensive Course Series on Applied Machine Learning in the Automotive Industry**

Welcome to the Auto Major Series - an intensive, hands-on course designed to teach advanced machine learning techniques through real-world automotive applications. This course combines theoretical foundations with practical implementation, covering the complete spectrum from advanced algorithms to production deployment, led by industry experts.

## 👥 Course Leadership

**Theory Lead**: Dr. Satya Jayadev  
**Hands-On Lead**: Aditya Ramesh Ganti

## 📚 Course Overview

This comprehensive course series provides participants with end-to-end machine learning expertise spanning multiple domains and technologies essential for modern automotive analytics:

### 🚀 Advanced Machine Learning Algorithms
Master cutting-edge algorithms specifically applied to automotive challenges:
- **XGBoost & LightGBM**: High-performance gradient boosting for warranty prediction and risk assessment
- **Anomaly Detection**: Identifying unusual patterns in vehicle performance and maintenance data
- **Clustering Techniques**: Customer segmentation, vehicle grouping, and pattern discovery
- **Real-world Applications**: Warranty cost prediction, failure pattern analysis, and predictive maintenance

### ⚙️ ML Deployment & Best Practices
Transform models from notebooks to production systems:
- **MLOps Fundamentals**: End-to-end pipeline automation and monitoring
- **CI/CD for ML**: Continuous integration and deployment strategies for model updates
- **Model Versioning**: Track, compare, and manage model iterations effectively
- **Cloud Platforms**: Hands-on experience with Azure ML and AWS SageMaker
- **Production Deployment**: Scalable, maintainable ML systems in cloud environments

### 🔤 Advanced Natural Language Processing
Apply NLP techniques to automotive text data:
- **LSTM Networks**: Sequential modeling for time-series analysis and text processing
- **Transformer Models**: State-of-the-art language models for automotive documentation analysis
- **Sentiment Analysis**: Customer feedback analysis and brand perception monitoring
- **Text Classification**: Automated categorization of service reports and warranty claims

### 👁️ Computer Vision for Quality Control
Implement image-based solutions for automotive quality assurance:
- **Image Classification**: Automated defect detection in automotive components
- **Quality Inspection Systems**: Real-time visual quality control implementation
- **Deep Learning for Vision**: Convolutional neural networks for automotive applications
- **Production Integration**: Deploying vision systems in manufacturing environments

### 🎯 Learning Objectives

By the end of this comprehensive course, participants will master:

**Technical Excellence:**
- Advanced ensemble methods (XGBoost, LightGBM) with hyperparameter optimization
- Production-ready ML deployment using MLOps best practices
- NLP applications for automotive text analysis and customer insights
- Computer vision systems for automated quality inspection
- Cloud-native ML solutions with proper monitoring and versioning

**Business Impact:**
- End-to-end ML pipeline design from problem definition to production
- ROI analysis and business value quantification for ML initiatives
- Automated quality control systems reducing manual inspection costs
- Predictive maintenance strategies minimizing downtime and costs
- Data-driven decision making frameworks for automotive operations

**Industry Expertise:**
- India-specific automotive challenges and environmental factor modeling
- Warranty cost optimization and risk assessment strategies
- Customer sentiment analysis and brand perception monitoring
- Manufacturing quality control automation
- Regulatory compliance and safety standard adherence

---

## 🔧 Foundation Project: Warranty Cost Repair Prediction

### Project Overview

The course begins with a comprehensive warranty cost prediction project for Auto Major vehicles in the Indian market, establishing the practical foundation for all advanced techniques covered throughout the series.

### 📊 Dataset Characteristics

**Synthetic Dataset Features:**
- **25,000 vehicle records** with complete warranty information
- **India-specific environmental factors**: Monsoon exposure, air pollution, dust levels
- **Comprehensive vehicle data**: Make, model, engine type, transmission, usage patterns
- **Regional variations**: Six major Indian regions with distinct characteristics
- **Component failure risks**: Engine, transmission, electrical, suspension, brake, AC systems
- **Target variables**: Repair costs (INR), claim counts, high-cost claim flags

**Key Business Metrics:**
- Average repair cost: ₹57,819
- High-cost claims (>₹75K): 22.8% of total claims
- Top 5% of claims account for 30.9% of total warranty costs
- Environmental severity impact: 1.4x cost difference between high/low severity regions

### 💻 Technical Implementation
#### 2. Exploratory Data Analysis (`notebooks/eda_and_data_preprocessing_notebook.ipynb`)

**Advanced Categorical Encoding:**
```python
# Ordinal Mappings for Naturally Ordered Variables
ordinal_mappings = {
    'monsoon_exposure': {'Minimal': 1, 'Low': 2, 'Medium': 3, 'High': 4},
    'air_pollution_level': {'Low': 1, 'Moderate': 2, 'High': 3, 'Severe': 4},
    'fuel_quality': {'Below_Standard': 1, 'Standard': 2, 'Premium': 3},
    'road_surface_type': {'Paved_Highway': 1, 'Paved_City': 2, 'Semi_Paved': 3, 'Gravel': 4, 'Dirt_Track': 5}
}
```

**Comprehensive Analysis Features:**
- Ordinal encoding for naturally ordered variables (pollution levels, road quality)
- One-hot encoding for nominal categorical variables
- Label encoding for high-cardinality features
- Cost distribution analysis with regional breakdowns
- Component failure risk assessment and environmental impact quantification

#### 3. Machine Learning Models (`notebooks/xgb_sup_lr_warr_cost_pred.ipynb`)

**Supervised Learning Implementation:**
- **XGBoost Regression**: Precise warranty repair cost prediction
- **XGBoost Classification**: High-cost claim identification
- **Advanced Feature Engineering**: Risk aggregations and interaction terms
- **Business Impact Analysis**: ROI calculations and cost savings projections

### 📈 Results and Business Impact

**Model Performance & Applications:**
1. **Warranty Cost Optimization**: 15-20% reduction in unexpected repair costs
2. **Proactive Maintenance**: Early identification of high-risk vehicles
3. **Regional Strategy**: Customized warranty terms based on environmental factors
4. **Inventory Management**: Optimized parts stocking based on failure predictions

### 🛠️ Technical Requirements

**Environment Setup:**
```bash
# Install dependencies using uv
uv sync

# Activate environment
source .venv/bin/activate
```

**Key Dependencies:**
- **Data Processing**: pandas, numpy, scipy
- **Machine Learning**: scikit-learn, xgboost, lightgbm
- **Visualization**: matplotlib, seaborn
- **Development**: jupyter, notebook

### 📁 Project Structure

```
warranty_cost_repair_prediction/
   data/
      clustering_and_anomaly_data/
         abc_warranty_anomaly_data.csv   # Anomaly detection dataset
         abc_warranty_clustering_data.csv # Clustering analysis dataset
      sup_lr_data/
         *.csv                           # Supervised learning datasets
   notebooks/
      # Core ML Notebooks
      eda_and_data_preprocessing_notebook.ipynb    # Comprehensive EDA
      xgb_sup_lr_warr_cost_pred.ipynb             # XGBoost supervised learning

      # Advanced Analytics Notebooks
      abc_anomaly_detection_notebook.ipynb        # Anomaly detection hands-on session
      clustering_analysis_enhanced.ipynb          # Advanced clustering analysis
      clustering_analysis_notebook.ipynb          # Basic clustering analysis
```

### 🚀 Getting Started

1. **Clone and Setup**:
   ```bash
   git clone <repository-url>
   cd auto-major-adv-ml-series
   uv sync
   ```

2. **Run Analysis**:
   ```bash
   jupyter notebook
   # Navigate to notebooks/ and start with EDA notebook
   ```

---

## 📝 Course Logs

### Current Status

**Foundation Project - Warranty Cost Prediction:**
- ✅ Enhanced EDA notebook with ordinal encoding for categorical variables
- ✅ Implemented XGBoost supervised learning for both regression and classification
- ✅ Added comprehensive business impact analysis
- ✅ Created India-specific environmental factor modeling

**Advanced Analytics - Recently Completed:**
- ✅ Anomaly detection notebook with PyOD library implementation
- ✅ Comprehensive anomaly vs outlier distinction analysis
- ✅ Enhanced clustering analysis with multiple algorithms
- ✅ Reorganized data structure with specialized subdirectories

**MLOps & Production Deployment - Recently Completed:**
- ✅ Complete MLOps workshop with MLflow tracking and model registry
- ✅ XGBoost and LightGBM implementation with 5-fold cross-validation
- ✅ Automated experiment tracking with per-fold metrics visualization
- ✅ Model comparison and selection framework (XGBoost vs LightGBM)
- ✅ Production-ready model deployment simulation

**Upcoming Development:**
- 📋 Cloud deployment with AWS SageMaker/Azure ML
- 📋 NLP components for automotive text analysis
- 📋 Computer vision modules for quality inspection
- 📋 Advanced hyperparameter tuning with Hyperopt/Optuna

### Technical Milestones

| Component | Feature | Status |
|-----------|---------|---------|
| Foundation | Ordinal encoding implementation | ✅ Complete |
| Foundation | XGBoost regression model | ✅ Complete |
| Foundation | XGBoost classification model | ✅ Complete |
| Foundation | Business impact analysis | ✅ Complete |
| Advanced ML | Anomaly detection with PyOD | ✅ Complete |
| Advanced ML | Clustering analysis | ✅ Complete |
| Advanced ML | LightGBM implementation | ✅ Complete |
| MLOps | MLflow experiment tracking | ✅ Complete |
| MLOps | K-fold cross-validation | ✅ Complete |
| MLOps | Model registry & versioning | ✅ Complete |
| MLOps | Auto-generated visualizations | ✅ Complete |
| MLOps | Cloud deployment (AWS/Azure) | 📋 Planned |
| NLP | Text analysis modules | 📋 Planned |
| Vision | Quality inspection system | 📋 Planned |

---

## 🚀 MLOps Project: Predictive Maintenance with MLflow

### Project Overview

A comprehensive hands-on MLOps workshop demonstrating end-to-end machine learning operations for manufacturing predictive maintenance, featuring advanced experiment tracking, model versioning, and production deployment strategies.

### 📊 Dataset & Problem

**Challenge:** Severe class imbalance (28.5:1 ratio) in manufacturing failure prediction
- **10,000 machine operation records** from Kaggle
- **3.39% failure rate** - extreme imbalance requiring specialized techniques
- **6 sensor features**: Temperature, RPM, torque, tool wear
- **Target**: Binary classification (failure vs normal operation)

**Key Insight:** Despite severe imbalance, Cohen's d analysis reveals strong feature separation (Torque d=0.788), enabling high-accuracy prediction with proper techniques.

### 💻 Technical Implementation

**Location:** `predictive_maintenance/`

#### Complete MLOps Workflow (`notebooks/mlops_complete_workshop.ipynb`)

**Environment-Agnostic Design:**
- Single `ENVIRONMENT` variable switches between Google Colab and local execution
- Automatic dependency management and path configuration
- Works seamlessly on both cloud and local development environments

**Key Features:**

1. **Advanced EDA & Feature Engineering:**
   - Cohen's d effect size analysis for feature quality assessment
   - Temperature gradients, mechanical stress indicators, power calculations
   - Wear intensity metrics and interaction features

2. **Multiple Model Training:**
   - **Baseline XGBoost**: Simple parameters for comparison baseline
   - **Optimized XGBoost**: Grid-searched hyperparameters
   - **XGBoost + 5-Fold CV**: Robust evaluation with variance estimates
   - **LightGBM + 5-Fold CV**: Algorithm comparison for data-driven selection

3. **MLflow Experiment Tracking:**
   ```python
   # Dataset versioning with digest tracking
   dataset = mlflow.data.from_pandas(df, source=path, name="predictive_maintenance")

   # Per-fold metrics logged as time-series (auto-generates charts)
   mlflow.log_metric("fold_f1_score", f1_value, step=fold_number)

   # Model registry with lifecycle management
   mlflow.register_model(model_uri, "predictive-maintenance-classifier")
   ```

4. **Automated Visualizations:**
   - MLflow UI automatically generates line charts from step-based metrics
   - No manual PNG creation required
   - Interactive comparison of XGBoost vs LightGBM across all folds

5. **Production Deployment Simulation:**
   - Model loading by version/stage
   - Signature validation for input schema
   - Multiple deployment options (local, REST API, SageMaker, Docker)

### 📈 Results & MLOps Achievements

**Model Performance:**
- **All metrics >80%**: Balanced accuracy ~91%, Precision ~85%, Recall ~84%, F1 ~84%
- **Robust across folds**: Low standard deviation indicates stable predictions
- **Production-ready**: Achieves business requirements despite 28.5:1 imbalance

**MLOps Capabilities Demonstrated:**
- ✅ **Experiment Tracking**: All parameters, metrics, and artifacts logged systematically
- ✅ **Dataset Versioning**: MD5 digest tracking for reproducibility
- ✅ **Model Versioning**: Centralized registry with version control
- ✅ **K-Fold Evaluation**: Per-fold metrics with automatic chart generation
- ✅ **Model Comparison**: Data-driven selection between algorithms
- ✅ **Lifecycle Management**: Staging → Production workflow
- ✅ **Artifact Storage**: Models, plots, metadata organized hierarchically

**Key Technical Highlights:**
- **XGBoost `scale_pos_weight`**: Solved extreme imbalance by weighting failure samples 28.5x
- **Cohen's d = 0.788**: Quantified strong feature separation despite class rarity
- **Step-based logging**: Enabled automatic MLflow UI chart generation
- **Stratified K-Fold**: Maintained class distribution across all folds
- **Environment agnostic**: Single codebase for Colab and local execution

### 🛠️ Running the Workshop

```bash
# Environment setup
uv sync
source .venv/bin/activate

# Launch Jupyter
jupyter notebook predictive_maintenance/notebooks/mlops_complete_workshop.ipynb

# Set ENVIRONMENT variable in notebook (Cell 2):
ENVIRONMENT = 'local'  # or 'colab'

# Run all cells to train 4 models and log to MLflow

# Launch MLflow UI to view experiments
mlflow ui --backend-store-uri predictive_maintenance/notebooks/mlruns --port 5000
# Open browser to http://localhost:5000
```

**MLflow UI Exploration:**
1. View all 4 experiment runs (baseline, optimized, XGBoost k-fold, LightGBM k-fold)
2. Click any run → Metrics → Click `fold_f1_score` → **Automatic line chart appears**
3. Select 2 runs → Click "Compare" → Side-by-side metrics and overlaid fold charts
4. Filter runs: `tags.model_type = 'xgboost'` or `metrics.cv_mean_f1_score > 0.84`
5. Download artifacts: confusion matrices, feature importance plots, trained models

### 📁 Project Structure

```
predictive_maintenance/
   data/
      predictive_maintenance.csv              # Kaggle dataset (10K records)
   notebooks/
      mlops_complete_workshop.ipynb           # Complete MLOps workshop (55 min)
      mlruns/                                 # MLflow tracking directory
         <experiment_id>/
            <run_id>/
               params/                        # Logged hyperparameters
               metrics/                       # Logged metrics (with step data)
               artifacts/                     # Models, plots, CSVs
               inputs/                        # Dataset version metadata
         models/                              # Model registry
            predictive-maintenance-classifier/
               version-1/                     # Registered model versions
```

### 🎓 Workshop Learning Objectives

**For Data Scientists:**
- Implement systematic experiment tracking with MLflow
- Use k-fold cross-validation for robust model evaluation
- Compare multiple algorithms objectively with metrics
- Understand Cohen's d for feature quality assessment
- Handle severe class imbalance with `scale_pos_weight`

**For ML Engineers:**
- Design environment-agnostic ML notebooks
- Implement model versioning and registry systems
- Create automated visualization pipelines
- Prepare models for production deployment
- Understand MLflow artifact storage architecture

**For MLOps Practitioners:**
- Build reproducible ML workflows
- Track dataset versions with digest hashing
- Manage model lifecycle stages (Staging → Production)
- Generate automatic experiment visualizations
- Deploy models with multiple strategies

---

## 🎯 Target Audience

This comprehensive course is designed for:

**Technical Professionals:**
- **Data Scientists** working in automotive or manufacturing industries
- **ML Engineers** focusing on production deployment and advanced algorithms
- **DevOps Engineers** interested in MLOps and automated deployment pipelines
- **Computer Vision Engineers** working on quality control and inspection systems

**Business Professionals:**
- **Business Analysts** seeking to understand advanced analytics and ROI measurement
- **Automotive Professionals** interested in data-driven decision making
- **Quality Control Managers** looking to implement automated inspection systems
- **Product Managers** overseeing ML initiatives in automotive companies

### Prerequisites
- Solid understanding of Python and pandas
- Fundamental knowledge of machine learning concepts
- Interest in automotive industry applications
- Experience with Jupyter notebooks
- Basic understanding of cloud platforms (for deployment modules)
- Familiarity with deep learning concepts (for NLP and computer vision modules)

### Support and Resources
- **Code Examples**: Fully documented notebooks with detailed explanations
- **Business Context**: Industry-specific insights and real-world applications
- **Best Practices**: Production-ready code and methodologies
- **Expert Guidance**: Theory from Dr. Jayadev, hands-on implementation from Aditya

---

## 📞 Contact and Support

**Course Leadership:**
- **Theory Lead**: Dr. Jayadev
- **Hands-On Lead**: Aditya Ramesh Ganti
  - Email: contactme@gadityaramesh.com
  - Professional Email: adityaramesh.g@gyandata.com

For technical questions, course content, or collaboration opportunities, please reach out to the respective leads based on your inquiry type.

---

*This course series represents a comprehensive journey through modern machine learning applications in the automotive industry, from foundational algorithms to advanced deployment strategies and specialized applications in NLP and computer vision.*
