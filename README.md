# pvrnn_sa
PyTorch code about PV-RNN (python)

The folder structure is below.  

1. “training/“  
  –”networks/”: Some python codes defining PV-RNN structures and functions (e.g., initialization, forward generation)  
  –“example/”: Codes for setting network hyperparameters and executing program (training).  
    ・“example_of_training_rnn_part.py”: Main code for running program  
    ・“network_config.yaml”: Hyperparameter setting  
    ・(“result_training/”, “trained_model/”): Results are saved here  
    ・“target/”: Target data (training data) is placed here  
    ・“dataset.py”, “utilities.py”: Some supplemental functions read by “example_of_training_rnn_part.py”  
    
2. “test/”  
  –”networks/”: Some python codes defining PV-RNN structures and functions (e.g., initialization, forward generation)  
  –“example/”: Codes for setting network parameters and executing program (online inference test).  
    ・“example_of_error_regression.py”: Main code for running program  
    ・“network_config.yaml”: Hyperparameter setting  
    ・(“result_ER/”): Results are saved here  
    ・“target/”: Target data (test data) is placed here  
    ・“error_regression.py”, “dataset.py”, “utilities.py”: Some supplemental functions read by “example_of_error_regression.py”  
    ・“generate*.npy”, “model*.npy”: Trained parameters read by “example_of_error_regression.py” 
