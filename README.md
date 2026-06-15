# India Demographics ML Dashboard [![Run Pytest Automation](https://github.com)](https://github.com/srinivasta/india-demographics-ml-dashboard)

### **Created by Srinivasta** 🚀

An interactive, data-driven analytics web application engineered by **Srinivasta** and deployed via the [Streamlit Community Cloud](https://india-demographics-ml-dashboard-wcou2wqfa4dxlv2me7bu2g.streamlit.app/)
). This platform visualizes historical census transitions across India's States and Union Territories from 1901 to 2026, alongside continuous Machine Learning projections extended up to 2036.

---

## 🚀 Key Features

* **Dynamic Regional Analytics**: Seamlessly toggle views between "India (Total Country)" and individual States/UTs using a reactive sidebar dropdown menu.
* **Dual-Axis Chronological Visualization**: Dual-axis charts built via Matplotlib that contrast annual event volumes (Births, Deaths, and Net Growth) with total cumulative population trends.
* **Hybrid Data Layering**: Integrates actual government baseline anchor markers with advanced machine learning forecast prediction modeling.
* **Custom Analysis Horizons**: Slide and pick distinct historical or futuristic timeline window ranges via a select-slider to slice metrics dynamically.
* **On-the-Fly Exports**: Instantly download your filtered data views directly as structured CSV spreadsheets.

---

## 🛠️ Machine Learning Architecture

The application uses an object-oriented regression pipeline to model historical patterns and project demographic shifts out to 2036:
1. **Feature Engineering**: Polynomial Features (`degree=1` with anchoring overrides) transform linear time vectors to capture accelerated, decelerated, and stabilizing non-linear demographic shifts.
2. **Predictive Modeling**: A regularized Ridge Regression algorithm balances model complexity, preventing over-fitting on highly sparse early-century documentation data.
3. **Pandas Processing**: Fully automated interpolation frameworks bridge data gaps between primary historical anchor census collection benchmarks.

---

## 🧪 Automated Testing Suite (`pytest`)

This repository features a comprehensive continuous integration (CI) pipeline powered by **GitHub Actions** and **Pytest**. Every code check-in triggers cloud validations to protect the mathematical and layout integrity of the system:

* **Data Integrity Tests**: Verifies schema definitions, checks for missing columns, and ensures data matrices contain zero empty nodes.
* **Mathematical Bound Constraints**: Confirms that population metrics never output impossible numbers (e.g., negative populations or negative births).
* **Historical Logic & Anomaly Checks**: Confirms data sequences line up chronologically between landmark census milestones and checks the 2020 COVID death rate anomaly scaling logic.
* **Streamlit UI Simulations**: Uses `streamlit.testing` to run headless browser simulations. This actively clicks sidebar components, triggers region dropdown changes (like selecting *Andaman and Nicobar Islands*), shifts timeline values, and asserts that UI rendering layers never crash.

---

## 📦 Project Directory Structure

```text
india-demographics-ml-dashboard/
├── .github/workflows/pytest.yml  # GitHub Actions CI cloud automation pipeline
├── app.py                         # Main Streamlit application executable script code
├── test_app.py                    # Pytest framework automated testing scripts
├── requirements.txt               # Python runtime package dependency checklist file
└── README.md                      # Project documentation and architectural layout summary
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

### 4. Run the Automated Tests
```bash
python -m pytest test_app.py -v
```

### 5. Fire Up the Dashboard App
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
pytest>=8.0.0
pytest-html
```

---

## 👤 Developer & Credits

* **Lead Architect & Developer**: Srinivasta
* **Application Hosting Platform**: Built using [Streamlit](https://india-demographics-ml-dashboard-wcou2wqfa4dxlv2me7bu2g.streamlit.app/) Framework
* **Data Sources**: Historical Census Reports of India & Projected Demographic Anchor Frameworks.

---

## 🛡️ License

This project is open-source and available under the **MIT License**.
