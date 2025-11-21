# College Artificial Intelligence 2025

## Overview

This repository contains a comprehensive collection of practical exercises and projects for an Artificial Intelligence course. The materials cover fundamental to advanced topics in AI, machine learning, and computer vision, organized into structured learning modules (jobsheets).

## Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Course Modules](#course-modules)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Assessments](#assessments)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

```
college-artificial-intelligence-2025/
├── jobsheet_01/          # Computer Vision - Face Detection
├── jobsheet_02/          # Python Fundamentals
├── jobsheet_03/          # Statistical Analysis with NumPy and SciPy
├── jobsheet_04/          # Data Processing with Pandas
├── jobsheet_05/          # Advanced Data Analysis
├── jobsheet_06/          # Data Visualization
├── jobsheet_07/          # Machine Learning Basics
├── jobsheet_08/          # Supervised Learning Algorithms
├── jobsheet_09/          # Classification Algorithms (SVM, SVR, Naive Bayes)
├── jobsheet_10/          # Clustering Analysis
├── UTS/                  # Midterm Examination
├── Akhir/                # Final Project
└── requirements.txt      # Python dependencies
```

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.8 or higher
- pip (Python package installer)
- Git
- A webcam (for computer vision exercises in jobsheet_01)
- Jupyter Notebook or JupyterLab (recommended for .ipynb files)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/setyanoegraha/college-artificial-intelligence-2025.git
cd college-artificial-intelligence-2025
```

### 2. Create a Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Course Modules

### Jobsheet 01: Computer Vision - Face Detection

Introduction to computer vision using OpenCV for real-time face and smile detection.

**Topics Covered:**
- Face detection using Haar Cascades
- Real-time webcam processing
- Smile detection algorithms

**Key Files:**
- `face_detector/` - Static image face detection
- `face_detector_webcam/` - Real-time face detection
- `face_and_smile_detector_webcam/` - Combined face and smile detection
- `smile_detector/` - Smile detection implementation

### Jobsheet 02: Python Fundamentals

Core Python programming concepts essential for AI development.

**Topics Covered:**
- Data types and variables
- Control structures
- Functions and modules
- Basic Python operations

### Jobsheet 03: Statistical Analysis

Statistical methods using NumPy and SciPy libraries.

**Topics Covered:**
- Central tendency (mean, median, mode)
- Statistical distributions
- Data analysis techniques
- NumPy array operations

### Jobsheet 04: Data Processing with Pandas

Data manipulation and processing using the Pandas library.

**Topics Covered:**
- Reading CSV and JSON files
- Data frame operations
- Data cleaning and preprocessing
- Data transformation techniques

### Jobsheet 05: Advanced Data Analysis

More advanced data analysis techniques and workflows.

**Topics Covered:**
- Complex data manipulations
- Data aggregation
- Time series analysis
- Advanced Pandas operations

### Jobsheet 06: Data Visualization

Creating meaningful visualizations to understand and present data.

**Topics Covered:**
- Matplotlib fundamentals
- Statistical plots
- Data storytelling
- Interactive visualizations

### Jobsheet 07: Machine Learning Basics

Introduction to machine learning concepts and algorithms.

**Topics Covered:**
- Supervised vs unsupervised learning
- Model training and evaluation
- Feature engineering
- Cross-validation techniques

### Jobsheet 08: Supervised Learning Algorithms

Implementation of key supervised learning algorithms.

**Topics Covered:**
- Decision Trees
- Linear Regression
- Logistic Regression
- Model evaluation metrics

**Key Subdirectories:**
- `latihan_1_decision_tree/` - Decision tree classification
- `latihan_2_linear_regression/` - Linear regression models
- `latihan_3_logistic_regression/` - Logistic regression implementation

### Jobsheet 09: Classification Algorithms

Advanced classification techniques and algorithms.

**Topics Covered:**
- Support Vector Machines (SVM)
- Support Vector Regression (SVR)
- Naive Bayes Classifier
- Model comparison and selection

**Key Subdirectories:**
- `1_ImplementasiSupportVectorMachineSVM/` - SVM implementation
- `2_ImplementasiSupportVectorRegression(SVR)/` - SVR implementation
- `3_ImplementasiNaiveBayes/` - Naive Bayes classifier

### Jobsheet 10: Clustering Analysis

Unsupervised learning and clustering techniques.

**Topics Covered:**
- K-Means clustering
- Hierarchical clustering
- Cluster evaluation
- Customer segmentation

## Usage

### Running Python Scripts

Navigate to the specific jobsheet directory and run the Python script:

```bash
cd jobsheet_01/face_detector_webcam
python face_detector_webcam.py
```

### Running Jupyter Notebooks

Launch Jupyter Notebook from the repository root:

```bash
jupyter notebook
```

Then navigate to the desired notebook file (`.ipynb`) and open it in your browser.

### Example: Face Detection

```bash
cd jobsheet_01/face_detector_webcam
python face_detector_webcam.py
```

Press 'q' to quit the webcam window.

### Example: Data Analysis

Open the relevant Jupyter notebook:

```bash
jupyter notebook jobsheet_05/latihan/latihan_1.ipynb
```

## Dependencies

The project uses the following major libraries:

- **OpenCV (opencv-python)**: Computer vision and image processing
- **NumPy**: Numerical computing and array operations
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical data visualization
- **Scikit-learn**: Machine learning algorithms and tools
- **SciPy**: Scientific computing and statistics
- **XGBoost**: Gradient boosting framework
- **Imbalanced-learn**: Handling imbalanced datasets

For a complete list, see [requirements.txt](requirements.txt).

## Assessments

### Midterm Examination (UTS)

Located in the `UTS/` directory, contains:
- Airbnb dataset analysis
- Data preprocessing and exploration
- Predictive modeling

### Final Project (Akhir)

Located in the `Akhir/` directory, contains:
- Patient data analysis
- Complete machine learning pipeline
- Final project implementation

## Project Organization

Each jobsheet follows a consistent structure:

```
jobsheet_XX/
├── latihan/           # Practice exercises
└── tugas_praktikum/   # Graded assignments
```

- **latihan**: Contains practice exercises to learn concepts
- **tugas_praktikum**: Contains practical assignments for assessment

## Tips for Success

1. **Complete exercises in order**: Each jobsheet builds upon previous knowledge
2. **Experiment with code**: Modify parameters and observe results
3. **Read error messages carefully**: They often contain helpful information
4. **Use virtual environments**: Keeps dependencies isolated and manageable
5. **Review documentation**: Refer to official library documentation when needed

## Troubleshooting

### Common Issues

**Webcam not detected:**
- Ensure your webcam is properly connected
- Check if other applications are using the webcam
- Verify camera permissions in your operating system

**Module not found errors:**
- Ensure virtual environment is activated
- Reinstall requirements: `pip install -r requirements.txt`
- Check Python version compatibility

**Jupyter Notebook issues:**
- Install Jupyter if not already installed: `pip install jupyter`
- Launch from the correct directory
- Clear kernel and restart if notebook becomes unresponsive

## Additional Resources

- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

## Contributing

This is an educational repository for course work. If you find any issues or have suggestions:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is created for educational purposes as part of a college Artificial Intelligence course.

## Contact

For questions or issues related to this repository, please contact the repository owner or create an issue in the GitHub repository.

---

Last updated: November 2025