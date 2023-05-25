# Analysis Pipeline README

Similarly to the experiment, the analysis pipeline is configured using a yaml file. You should check the following settings before running the analysis script:

1. `participants` - Before running the script you can change which participants you want to include (if there is not parity between this list and the files available in the chosen directory you will receive a warning upon the script’s completion). 
2. `dependent_variable` - This can either be set to `salience` or `relevance`. 
3. `root` - this is where the data and plots will be saved. You must ensure that the tar file you have downloaded from surf drive is in the same directory specified as the `root`.

You can also change the experiment settings (which should be the same as those used in the experiment), the exclusion criteria, and the plot colors.

Download the subject folder called “Real” ([https://surfdrive.surf.nl/files/index.php/s/w889ZUF8zwEsL98](https://surfdrive.surf.nl/files/index.php/s/gAY5xBrhKGnxpRc) - email z.b.nudelman at student dot vu dot nl for the password) and ensure it is in the folder you specified as the `root`. 

Then go into your terminal and run `python analysis_pipeline.py`