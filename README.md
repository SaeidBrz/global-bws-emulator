# BWS Emulator: Training, Evaluation, and Scenario Prediction

*Repository README for the final LightGBM training, independent evaluation, and 36-file scenario-prediction workflow*

**Associated study:** “Exploring Future Blue Water Scarcity Dynamics Across Climate and Water-Demand Scenarios Through a Machine-Learning Emulator.”

**Scope note:** This README documents BWS_emulator_training_prediction_final.py.

# 1. Workflow scope

The script performs the following tasks:

- loads the historical blue water scarcity (BWS) training dataset;

- constructs monthly spatial, climatic, seasonal, and water-demand predictors;

- divides grid cells into spatially grouped training, validation, and test subsets;

- trains the final LightGBM regression emulator with early stopping;

- evaluates the model on the independent test subset; and

- generates monthly BWS predictions for 36 future-scenario files.

# 2. Execution environment

The supported execution environment is Google Colab with Python 3 and Google Drive mounted at /content/drive.

Required Python packages:

- numpy

- pandas

- matplotlib

- seaborn

- tqdm

- scikit-learn

- lightgbm

- google-colab

Package versions are not pinned inside the script. The environment used for the archived release should therefore be preserved separately when the repository is finalized.

# 3. Required directory structure

The script expects the following structure in Google Drive:

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><blockquote>
<p>BWS_Emulator_Data/<br />
├── Training/<br />
│ └── training_data.csv<br />
├── Prediction_input/<br />
│ ├── Baseline/<br />
│ ├── Climate_Sensitivity/<br />
│ └── Demand_Sensitivity/<br />
└── Prediction_output/<br />
├── Baseline/<br />
├── Climate_Sensitivity/<br />
└── Demand_Sensitivity/</p>
</blockquote></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

PROJECT_ROOT is set automatically to \<Google Drive root\>/BWS_Emulator_Data. Folder names and capitalization must match exactly.

# 4. Input data

## 4.1 Historical training file

File:

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><blockquote>
<p>Training/training_data.csv</p>
</blockquote></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

The file must contain one row per unique grid cell and the following columns:

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><blockquote>
<p>lon, lat<br />
bws_01 ... bws_12<br />
wd_01 ... wd_12<br />
precipitation_01 ... precipitation_12<br />
Temperature01 ... Temperature12</p>
</blockquote></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

Column names are case-sensitive. Historical temperature columns use Temperature01 to Temperature12, without an underscore before the month number.

## 4.2 Scenario-prediction files

Each folder under Prediction_input/ must contain the same 12 files:

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><blockquote>
<p>SSP1_2021_2040.csv SSP1_2041_2060.csv<br />
SSP1_2061_2080.csv SSP1_2081_2100.csv<br />
SSP2_2021_2040.csv SSP2_2041_2060.csv<br />
SSP2_2061_2080.csv SSP2_2081_2100.csv<br />
SSP5_2021_2040.csv SSP5_2041_2060.csv<br />
SSP5_2061_2080.csv SSP5_2081_2100.csv</p>
</blockquote></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

Each file must contain one row per unique grid cell and these columns:

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><blockquote>
<p>lon, lat<br />
wd_01 ... wd_12<br />
Temperature_01 ... Temperature_12<br />
Precipitation_01 ... Precipitation_12</p>
</blockquote></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

Scenario temperature and precipitation columns use an underscore before the month number. This naming differs from the historical training file and must be preserved exactly.

# 5. Model construction and evaluation

Each historical grid cell is converted into 12 monthly records. The model inputs include:

- water demand, precipitation, and temperature;

- hemisphere-aware season;

- sine and cosine encodings of calendar month;

- three-dimensional Cartesian coordinates derived from longitude and latitude; and

- the spatial and seasonal interaction terms defined in the script.

All 12 monthly records from the same grid cell receive the same group identifier. GroupShuffleSplit with random_state = 42 is used to create approximately 60% training, 20% validation, and 20% independent test subsets. Exact grid-cell groups cannot occur in more than one subset.

The final LGBMRegressor is trained with validation-based early stopping. Test predictions are evaluated without clipping or other post-processing. The script reports MSE, RMSE, MAE, R², NMSE, NSE, and PBIAS. Metrics and training history are printed during execution and are not automatically written to separate files.

# 6. Running the workflow

1.  Place the training and scenario files in the directory structure above.

2.  Open BWS_emulator_training_prediction_final.py in Google Colab.

3.  Run the script from beginning to end in one session.

4.  Confirm that the training-data and scenario-folder checks pass.

5.  Confirm that the final message reports 36 completed scenario predictions.

The script processes scenario files sequentially in this category order:

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><blockquote>
<p>Baseline<br />
Climate_Sensitivity<br />
Demand_Sensitivity</p>
</blockquote></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

# 7. Prediction outputs

Predictions are written to:

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><blockquote>
<p>Prediction_output/&lt;category&gt;/&lt;same_input_filename&gt;.csv</p>
</blockquote></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

Each output contains:

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><blockquote>
<p>lon, lat, bws_01, bws_02, ..., bws_12</p>
</blockquote></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

The output preserves the input row count, coordinate values, and coordinate order. Negative future BWS predictions are replaced with zero because BWS cannot be negative. No upper clipping is applied. Exactly 36 output CSV files must be generated; otherwise the workflow stops with an error.

# 8. Built-in integrity checks

- all required training and prediction columns are present;

- training and prediction files contain no duplicate lon/lat locations;

- grid-cell groups do not overlap among training, validation, and test subsets;

- prediction features match the training feature names and order;

- every output contains the expected 12 monthly BWS columns;

- output row counts match their corresponding inputs; and

- output coordinate order is preserved.

# 9. Interpretation

The model is a predictive emulator of BWS fields generated by the reference modelling workflow. Its outputs are conditional on the historical training data, engineered predictors, and scenario inputs supplied to the script. The workflow supports predictive emulation and scenario comparison and should not be interpreted as causal attribution of climate, water demand, location, or their interactions.
