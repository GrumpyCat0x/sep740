SEP740 Final Project: Stock Price Prediction Using LSTM
=======================================================

Project Overview
----------------
This project implements a Long Short-Term Memory (LSTM) neural network to
predict the next-day closing price of Apple Inc. (AAPL) using historical
Yahoo Finance data. The model is trained with PyTorch and evaluated against
a naive "tomorrow equals today" baseline. The implementation includes data
preprocessing, feature engineering, model training, evaluation, and result
visualization.

Repository Layout
-----------------
finalProject/
  README.txt
  requirements.txt
  aapl_yahoo_2015_2025.csv
  src/
    train_lstm.py
  outputs/
    metrics.json
    training_loss.png
    predictions.png
    predictions_zoomed.png
    residuals.png
  reports/
    detailed_report.docx
    ieee_report.pdf

Dataset
-------
The project uses historical daily stock prices of Apple Inc. (AAPL).

The training script can either:

1. Automatically download data from Yahoo Finance, or

2. Reuse the provided CSV file:

   aapl_yahoo_2015_2025.csv

Run the Project
---------------
To download data automatically:

  python src/train_lstm.py

To reproduce the reported results using the provided dataset:

  python src/train_lstm.py --csv aapl_yahoo_2015_2025.csv --epochs 20 --seed 740

Setup
-----
1. Create and activate a Python environment.

2. Install the required packages:

   pip install -r requirements.txt

3. Run the training script:

   cd finalProject
   python src/train_lstm.py --csv aapl_yahoo_2015_2025.csv --epochs 20 --seed 740

Expected Outputs
----------------
After execution, the outputs/ directory will contain:

  outputs/metrics.json
      Model evaluation metrics

  outputs/training_loss.png
      Training and validation loss curves

  outputs/predictions.png
      Actual vs. predicted closing prices

  outputs/predictions_zoomed.png
      Zoomed prediction plot for recent test samples

  outputs/residuals.png
      Distribution of prediction residuals

Implementation Notes
--------------------
- Framework: PyTorch
- Dataset: Yahoo Finance (AAPL)
- Input features:
    Open, High, Low, Close, Adj Close, Volume,
    Log_Return, MA5_Ratio, MA20_Ratio,
    Volatility20, Volume_Change

- Prediction target:
    Next-day log return, converted back to closing price for evaluation.

- Sequence length:
    60 trading days

- Data split:
    Chronological split with 80% training/validation
    and 20% testing.

- Feature scaling:
    StandardScaler fitted on the pre-test training period.

- Model:
    Two-layer LSTM with a fully connected regression head.

- Optimizer:
    Adam

- Loss function:
    Huber Loss (delta = 1.0)

- Learning rate:
    0.0005

- Early stopping:
    Patience = 5 with restoration of the best validation model.

Evaluation Metrics
------------------
The following metrics are reported:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Squared Error (MSE)
- Mean Absolute Percentage Error (MAPE)
- R-squared (R²)
- Directional Accuracy

Academic Integrity
------------------
All source code in src/train_lstm.py was developed for the SEP740 Final
Project. Public Python libraries are used through their official APIs. The
historical stock price data are obtained from Yahoo Finance using the
yfinance package.