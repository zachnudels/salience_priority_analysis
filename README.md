# Analysis Pipeline README

## Installation

The best way to run any Python package is using a (virtual) Python environment since 
many packages require different dependency versions. Probably the easiest way to 
install the required environment is with Anaconda. Anaconda is a science platform
for Python and R which makes managing such environments a bit easier. Included in 
the Anaconda download is the Spyder IDE which you can use to edit the python
files in this package.

Once you have downloaded Anaconda, you can perform the following steps
in the Anaconda Prompt to create the environment and install the required 
dependencies:
1. Create a new environment (called salience_priority_analysis) with 
    
    `conda create --name salience_priority_analysis python=3.9`
2. Activate the environment with
    
    `conda activate salience_priority_analysis`
3. Change directory to the root of this repo
    
    `cd <your_directory_path>/salience_priority_analysis`
4. Install the required packages
    
    `pip install -r requirements.txt`

** Note that you only need to do the above once! Once you have set up your environment it is set up forever!


## Usage

### Pipeline Settings
Similarly to the experiment, the analysis pipeline is configured using a yaml file 
`analysis_pipeline.yml`

You should check the following settings before running the analysis script:

1. `dependent_variables` - Will usually be `[salience, relevance]`. 
2. `root` - this is where the data and plots will be saved. You must ensure that 
the tar (or zip) file you have downloaded from surf drive is in the same directory 
specified as the `root`.

You can also change the experiment settings (which should be the same as those 
used in the experiment), the exclusion criteria, and the plot colors.


### Download the Data
Download the subject folder called “Real” 
([https://surfdrive.surf.nl/files/index.php/s/w889ZUF8zwEsL98](https://surfdrive.surf.nl/files/index.php/s/gAY5xBrhKGnxpRc) -
email z dot b dot nudelman at student dot vu dot nl for the password)
Ensure the downloaded (zip or tar) file is in the folder you specified as the 
`root` in `analysis_settings.yml`


### Run the Pipeline
1. Open Anaconda Prompt
2. Change to the directory containing the file `analysis_pipeline.py`
3. Activate your environment with the installed dependencies 
with `conda activate salience_priority_analysis`
4. Run `python analysis_pipeline.py`


### Expected Output
The pipeline should create the following folders under `root`:
1. `parsed_data`: Dataframes generated from eyetracking parsing package `EyeParser` 
2. `clean_data`: Cleaned `parsed_data` dataframes (excludes invalid trails, etc.)
3. `ANOVA`: For each participant, a `.csv` file containing relevant information 
needed to run an ANOVA.
4. `tables`: `.csv` files containing the trial balances across SOA and saliency per
participant
5. `figs`: Generated figures
6. `SMARTdata.p`: This file is used for the SMART analysis
