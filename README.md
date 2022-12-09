# DSCI 601 Sample Code
This is a sample of the code that has been worked on this semster. It is a part of the investigation into the viability of a selection of CMAB algorithms (and how they are more effective as compared to MAB algorithms in certain situations)

All the code is provided in a single jupyter notebook which can be executed once all the needed packages have been installed. The list of needed packages can be found in the "requierments.txt" file which contains a list of the pip names of the packages. 

The packages are as follows:
- pandas                | for dataset management 
- numpy                 | for quick array math and management
- sklearn               | for machine learning
- contextualbandits     | for bandit setup and training
- matplotlib            | for data visulization
- seaborn               | for data visulization
- tqdm                  | for streamlining looped data streaming

Once the packages have been installed in your virtual environment the script can be run by clciking on the ⏩  button.The code will execute cell by cell, first reading in the data from the provided dataset, scrambling the built data then testing the various CMAB algorithms on that data. After the CMAB algorithm testing the data is then also run on regular MAB algorithms to verify that contex did help the algorithms make better decisions