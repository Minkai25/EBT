# Building TRM dataset
When building the TRM dataset, you'll want to first zip the kaggle/combined directory from the TRM directory and unzip it in data/arc/raw-data,
then run this command after changing the output directory name
python -m data.arc.trm.build_arc_dataset     --input-file-prefix data/arc/raw-data/combined/arc-agi     --output-dir data/arc1concept-aug-10     --subsets training evaluation concept     --test-set-name 
evaluation

# Dataset naming
trm_small and trm_test are smaller versions of the dataset I made to debug. 
trm is the larger version of these datasets, while trm_original preserves the 
data-streaming from the original TRM repo. 

# Model Creation
I currently hardcode in the model parameters in base_model_trainer.py. 


