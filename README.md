# 📊 India State & UT Demographic Dashboard (1901–2036)
### *Created by Srinivasta*

An interactive, data-driven Streamlit analytics web application engineered by Srinivasta. This platform visualizes historical census transitions across India's States and Union Territories from 1901 to 2026, alongside continuous Machine Learning projections extended up to 2036.

---

## 🚀 Key Features

* **Dynamic Regional Analytics**: Seamlessly toggle views between "India (Total Country)" and individual States/UTs using a reactive sidebar interface.
* **Dual-Axis Chronological Visualization**: Dual-axis charts built via Matplotlib that contrast annual event volumes (Births, Deaths, and Net Growth) with total cumulative population trends.
* **Hybrid Data Layering**: Integrates actual government baseline anchor markers with advanced machine learning forecast projection modeling.
* **Custom Analysis Horizons**: Slide and pick distinct historical or futuristic timeline window ranges (1901–2036) to slice metrics dynamically.
* **On-the-Fly Exports**: Instantly download your filtered data views directly as structured CSV spreadsheets.

---

## 🛠️ Machine Learning Architecture

The application uses an object-oriented regression pipeline to model historical patterns and project demographic shifts out to 2036:
1. **Feature Engineering**: Polynomial Features (`degree=2`) transform linear time vectors to capture accelerated, decelerated, and stabilizing non-linear demographic shifts.
2. **Predictive Modeling**: A regularized Ridge Regression algorithm balances model complexity, preventing over-fitting on highly sparse early-century documentation data.
3. **Pandas Processing**: Fully automated interpolation frameworks bridge data gaps between primary historical anchor census collection benchmarks.

---

## 📦 Project Directory Structure

```text
india-demographics-ml-dashboard/
├── app.py                # Main Streamlit application executable script code
├── requirements.txt      # Python runtime package dependency checklist file
└── README.md             # Project documentation and architectural layout summary
```

---

## ⚙️ Installation & Local Setup

Ensure you have Python 3.9 or higher installed on your computer. Follow these steps to run the application locally:

### 1. Clone the Repository
```bash
git clone https://github.com
cd india-demographics-ml-dashboard
```

### 2. Set Up a Virtual Environment
```bash
# Create environment
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Fire Up the Dashboard App
```bash
streamlit run app.py
```

---

## 📋 Package Dependencies (`requirements.txt`)

The core engine relies on the following stable framework distributions:
```text
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.2.0
```

---

## 👤 Developer & Credits

* **Lead Architect**: Srinivasta
* **Data Sources**: Historical Census Reports of India & Projected Demographic Anchor Frameworks.

---

## 🛡️ License

This project is open-source and available under the **MIT License**.
