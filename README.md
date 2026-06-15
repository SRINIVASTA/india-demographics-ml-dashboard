# India Demographics ML Dashboard [![Run Pytest Automation](https://github.com)](https://github.com/srinivasta/india-demographics-ml-dashboard)

### **Created by Srinivasta** 🚀

An interactive, data-driven analytics web application engineered by **Srinivasta** and deployed via the [Streamlit Community Cloud](https://india-demographics-ml-dashboard-wcou2wqfa4dxlv2me7bu2g.streamlit.app/). This platform visualizes historical census transitions across India's States and Union Territories from 1901 to 2026, alongside continuous Machine Learning projections extended up to 2036.

Check out our real-time [Live Automated Quality Report](https://github.io) to view the passing machine learning and user-interface simulation test suites.

---

## 🚀 Key Features

* **Dynamic Regional Analytics**: Seamlessly toggle views between "India (Total Country)" and individual States/UTs using a reactive sidebar dropdown menu.
* **Ensemble Model Comparison**: Swap between multiple regression algorithms (**Ridge, Lasso, and Linear Regression**) from the control panel to evaluate predictive variance across projection matrices.
* **Uncertainty Risk Shading**: Visualizes **95% Statistical Confidence Bands** around upcoming timeline projections, mapping prediction spread as it moves out toward 2036.
* **Dual-Axis Chronological Visualization**: Dual-axis charts built via Matplotlib that contrast annual event volumes (Births, Deaths, and Net Growth) with total cumulative population trends.
* **On-the-Fly Exports**: Instantly download your filtered data views directly as structured CSV spreadsheets.

---

## 🛠️ Machine Learning Architecture

The application uses an object-oriented regression pipeline to model historical patterns and project demographic shifts out to 2036:
1. **Feature Engineering**: Polynomial Features transform linear time vectors to capture accelerated, decelerated, and stabilizing non-linear demographic shifts.
2. **Predictive Modeling**: Incorporates an ensemble strategy utilizing **Ridge Regression**, **Lasso**, or **Ordinary Least Squares (OLS)** models to let users trace changes in baseline fit.
3. **Statistical Uncertainty**: Automatically computes historical residual standard deviations (`sigma`) to dynamically plot error bounds expanding into the future.
4. **Pandas Processing**: Fully automated interpolation frameworks bridge data gaps between primary historical anchor census collection benchmarks.

---

## 🧪 Automated Testing Suite (`pytest`)

This repository features a comprehensive continuous integration (CI) pipeline powered by **GitHub Actions** and **Pytest**. Every code check-in triggers cloud validations to protect the mathematical and layout integrity of the system:

* **In-App Testing Dashboard**: Features a native QA suite expander widget directly inside the web interface allowing users to run live headless validations with a single click.
* **Data Integrity Tests**: Verifies schema definitions, checks for missing columns, and ensures data matrices contain zero empty nodes.
* **Mathematical Bound Constraints**: Confirms that population metrics never output impossible numbers (e.g., negative populations) and checks that confidence boundaries logically frame predicted target metrics.
* **Historical Logic & Anomaly Checks**: Confirms data sequences line up chronologically between landmark census milestones and checks the 2020 COVID death rate anomaly scaling logic.

---

## 📦 Project Directory Structure

```text
india-demographics-ml-dashboard/
├── .github/workflows/pytest.yml  # GitHub Actions CI cloud automation pipeline
├── app.py                         # Main Streamlit user interface view layer
├── demographics_engine.py         # Isolated backend data processing & ML model pipeline
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

### 4. Run the Automated Tests Locally
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
