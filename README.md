# SpaceX Falcon 9 Landing Predictor

This project is a Data Science and Machine Learning application that predicts whether a SpaceX Falcon 9 booster will successfully land. It scrapes historical launch data from Wikipedia, processes the data, trains a Random Forest Classifier model, and provides a web-based interactive dashboard built with Dash to visualize the results and make predictions based on custom parameters.

## Features

- **Data Extraction**: Automatically scrapes Falcon 9 and Falcon Heavy launch data from Wikipedia.
- **Data Cleaning & Preprocessing**: Cleans and filters the data to extract relevant features like `Payload mass (kg)`, `Orbit`, and `Launch site`.
- **Machine Learning Model**: Trains a Random Forest Classifier to predict the success of a booster landing. The trained model and label encoders are saved using `joblib`.
- **Interactive Dashboard**: A Dash application (`app_spacex.py`) that includes:
  - **Launch Analytics**: Interactive visualizations showing landings by launch site, orbit, and payload mass vs landing success using Plotly.
  - **Custom Prediction Tool**: Allows users to input custom mission parameters (Payload Mass constrained between 0 and 25,000 kg, Orbit, Launch Site) to predict landing success in real-time. Inputs outside this bracket will indicate that the prediction cannot be determined.

## Project Structure

- `train_spacex.py`: Script to scrape data, preprocess it, train the Random Forest model, and save the artifacts (`spacex_model.joblib`, `falcon_9_filtered_data.csv`).
- `app_spacex.py`: The Dash web application that loads the trained model and provides the interactive user interface.
- `*.csv`: Cleaned datasets generated during the training phase.
- `*.joblib`: Saved machine learning models and encoders.

## Usage

1. **Train the Model**: Run `train_spacex.py` to extract the latest data and train the model.
   ```bash
   python train_spacex.py
   ```
2. **Run the Dashboard**: Run `app_spacex.py` to start the interactive web application.
   ```bash
   python app_spacex.py
   ```
   Navigate to the local server address (usually `http://127.0.1.1:8050/`) in your web browser.

## Requirements
- Python 3.x
- pandas
- numpy
- requests
- scikit-learn
- joblib
- dash
- plotly
- matplotlib
- seaborn
- lxml (for pandas to parse HTML)
