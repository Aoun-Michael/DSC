=============================================================================
PROJECT: Predictive Modeling for Donor Campaign Selection
=============================================================================

AUTHOR: Michael Aoun

-----------------------------------------------------------------------------
1. PROJECT OVERVIEW
-----------------------------------------------------------------------------
This project applies predictive modeling techniques to optimize donor 
selection for fundraising campaigns. By analyzing past donor behavior, 
gift history, and campaign data, the model identifies the most likely 
contributors for future campaigns (specifically targeting Campaign 7362), 
aiming to maximize ROI and improve campaign efficiency.

-----------------------------------------------------------------------------
2. DIRECTORY STRUCTURE
-----------------------------------------------------------------------------
DSC Project/
│
├── data/                               # Dataset directory
│   ├── campaigns.csv                   # Historical campaign metadata
│   ├── donors.csv                      # Donor demographic and profile data
│   ├── gifts.csv                       # Historical transaction/donation records
│   ├── selection campaign 6169.csv     # Target/selection data for campaign 6169
│   ├── selection campaign 7244.csv     # Target/selection data for campaign 7244
│   ├── selection campaign 7362.csv     # Target/selection data for campaign 7362
│   └── Scored_Campaign_7362_Final.csv  # Final model predictions/scores
│
├── docs/                               # Project documentation
│   ├── Business Case.xlsx              # Financial analysis and ROI projections
│   ├── DSC Presentation.pdf            # Slide deck summarizing findings
│   ├── Project Extra Info.pdf          # Additional context/data dictionary
│   └── Project Info.pdf                # Core project requirements/prompt
│
├── src/                                # Source code
│   ├── DSC Notebook.ipynb              # Main Jupyter Notebook with EDA & modeling
│   └── dsc_helper.py                   # Custom Python functions for data processing
│
└── README.txt                          # This file

-----------------------------------------------------------------------------
3. USAGE INSTRUCTIONS
-----------------------------------------------------------------------------
1. Helper Functions: 
   The `src/dsc_helper.py` file contains custom utility functions used for 
   cleaning data and evaluating the models. Ensure this file remains in the 
   `src/` directory so the notebook can import it.

2. Running the Analysis:
   Navigate to the `src/` directory and open `DSC Notebook.ipynb`. 
   Run the cells sequentially to reproduce the exploratory data analysis (EDA), 
   feature engineering, model training, and evaluation.

3. Final Output:
   The final scored predictions for the target campaign are outputted to 
   `data/Scored_Campaign_7362_Final.csv`.