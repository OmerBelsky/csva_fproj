# csva_fproj
train_models_combined.py is the main file. it runs the models for a certain amount of epochs and logs metrics with wandb.
in greedy_hyperp_optim.py we find the best hyper parameters for the model.
The rest of the files are used to generate the data.
Part of the preprocessing was actually aggregating the data
into a single file to save time when creating the DataLoader.
