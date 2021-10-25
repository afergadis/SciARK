# Datasets

## SciARK
The `SciARK.json` file is the complete dataset.
The other directories have the dataset splits used in the paper for the 
various experiments.

## AbstRCT
*"The [AbstRCT](https://gitlab.com/tomaye/abstrct/) dataset consists of 
randomized controlled trials retrieved from the MEDLINE database via PubMed
search. The trials are annotated with argument components and argumentative
relations.*"

This dataset is converted in order to be read and used from our models.

# Folds
The `SciARK` dataset is split into 10 folds with training, development and test
sets. Each directory contains one fold with the three dataset splits.

# Domains
Each directory has an SDG policy domain as test set. The remaining domains 
are merged and split into training and development set. We use those 
directories for the *cross-domain* experiment.