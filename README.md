# Valorant Agent Assistant

A Machine Learning-powered training tool for Valorant players. Built with Python, scikit-learn, and real game data.

## Overview

This project provides two workflows to help players improve:

1. **Weapon Recommender** - Uses a Decision Tree Classifier to analyze your playstyle and recommend optimal weapons
2. **Strategy Quiz** - Tests tactical decision-making with 180 map-specific scenarios

Two interface options are available: a desktop GUI (Tkinter) and a modern web app (Streamlit).

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Data Files](#data-files)
- [Screenshots](#screenshots)
- [License](#license)

---

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- streamlit (for web version)
- plotly (for web version)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/valorant-agent.git
cd valorant-agent

# Install dependencies
pip install pandas numpy scikit-learn matplotlib streamlit plotly
```

---

## Usage

### Desktop Application (Tkinter)

```bash
python valorant_agent.py
```

### Web Application (Streamlit)

```bash
streamlit run valorant_streamlit.py
```

The web app will open at `http://localhost:8501`

---

## Project Structure

```
valorant-agent/
├── valorant_agent.py        # Tkinter desktop application
├── valorant_streamlit.py    # Streamlit web application
├── weapons.csv              # Weapon database (19 weapons)
├── valorant_strategies.csv  # Quiz scenarios (180 questions)
├── agent.py                 # Reference implementation (cars model)
├── cars.csv                 # Reference data
└── README.md
```

---

## How It Works

### Weapon Recommender

The system uses a **Decision Tree Classifier** trained on synthetic player profiles. The model analyzes 7 features:

| Feature | Description |
|---------|-------------|
| Rank | Player skill tier (Iron to Radiant) |
| Sensitivity | Mouse sensitivity preference |
| Playstyle | Aggressive, passive, or balanced |
| Preferred Range | Close, medium, or long engagements |
| Aim Style | Flick, tracking, or spray control |
| Headshot Confidence | Self-reported headshot accuracy |
| Budget | Economy management habits |

The classifier predicts the optimal weapon from 19 options across 7 categories (Sidearms, SMGs, Shotguns, Rifles, Snipers, Machine Guns, Melee).

Training process:
1. Generate synthetic training data (20 profiles per weapon)
2. Encode weapon names with LabelEncoder
3. Split data 70/30 for training/testing
4. Train DecisionTreeClassifier (max_depth=8)
5. Evaluate accuracy on test set

### Strategy Quiz

The quiz pulls from a CSV database containing 180 tactical scenarios covering:

- 10 maps (Bind, Haven, Split, Ascent, Icebox, Breeze, Pearl, Fracture, Lotus, Sunset)
- Attack and defense sides
- 20 unique situation types (rushes, clutches, economy rounds, etc.)

Each question presents:
- Map and location
- Current side (attacking/defending)
- Situation description
- 4 possible actions

Performance is scored on accuracy and response time, with ranks from Iron to Radiant.

---

## Data Files

### weapons.csv

Contains all 19 Valorant weapons with stats:

- Weapon name
- Category
- Cost (credits)
- Fire mode
- Fire rate (rounds/sec)
- Magazine size
- Damage values (head/body/leg at 0-30m)
- Reload time
- Special notes

### valorant_strategies.csv

Contains 180 quiz questions with columns:

- Map
- Side (A=Attack, D=Defense)
- Location
- Situation
- Option1-4
- CorrectOption (index 0-3)
- Explanation

---

## Architecture

This project follows the same Decision Tree architecture as the reference `agent.py` (cars model):

```
Input Features -> LabelEncoder -> train_test_split -> DecisionTreeClassifier -> Prediction
                                        |
                                   Accuracy Score
                                        |
                                   plot_tree Visualization
```

Key components:
- `sklearn.tree.DecisionTreeClassifier` for classification
- `sklearn.preprocessing.LabelEncoder` for target encoding
- `sklearn.model_selection.train_test_split` for evaluation
- `sklearn.tree.plot_tree` for visualization
- `matplotlib` + `FigureCanvasTkAgg` for embedding plots in Tkinter

---

## Features

### Tkinter Application
- Valorant-themed dark UI (red/black color scheme)
- 7-question weapon profiling wizard
- 10-question strategy quiz with timer
- Decision tree visualization window
- Text-based rule export

### Streamlit Application
- Responsive web interface
- Interactive decision tree viewer with adjustable depth
- Feature importance chart
- Weapon database browser with filtering
- Real-time quiz scoring

---

## License

MIT License - feel free to use and modify.

---

## Author

Mohamed aziz ghazouani

---

## Acknowledgments

- Riot Games for Valorant
- scikit-learn documentation
