# pvrnn_sa
PyTorch code about PV-RNN (python)
1. System requirements (tested environment)  
    -OS: Ubuntu18  
    -Python3.7  
    -Pytorch (1.7)  
    -Computer platform: CPU  
  
2. Installation guide  
  If you have established Python environment, you may only have to install Pytorch according to instructions on the official website (https://pytorch.org/).
As needed, please install python packages such as matplotlib, glob, numpy, yaml, os, sys, time, and seaborn.
The installs may take no more than half-hour.

3. Demo and Instructions  
  The folder structure is below. 

“training/“  
    –”networks/”: Some python codes defining PV-RNN structures and functions (e.g., initialization, forward generation)  
    –“example/”: Codes for setting network hyperparameters and executing program (training).  
        ・“example_of_training_rnn_part.py”: Main code for running program  
        ・“network_config.yaml”: Hyperparameter setting  
        ・“result_training/”, “trained_model/”: Results are saved here  
        ・“target/”: Target data is placed here (target0000.txt ~ target0023: Self-produced condition, target0024.txt ~ target0047: Externally produced condition)  
        ・“dataset.py”, “utilities.py”: Some supplemental functions read by “example_of_training_rnn_part.py”  
    
“test/”  
    –”networks/”: Some python codes defining PV-RNN structures and functions (e.g., initialization, forward generation)  
    –“example/”: Codes for setting network parameters and executing program (online inference test).  
        ・“example_of_error_regression.py”: Main code for running program  
        ・“network_config.yaml”: Hyperparameter setting  
        ・“result_ER/”: Results are saved here  
        ・“target/”: Target data (test data) is placed here  
        ・“error_regression.py”, “dataset.py”, “utilities.py”: Some supplemental functions read by “example_of_error_regression.py”  
        ・“generate*.npy”, “model*.npy”: Trained parameters read by “example_of_error_regression.py” 

3-1. Learning experiment (It may take about one day to train one neural network.)
        
        cd /training/example/
        python example_of_training_rnn_part.py
        #Trained model and reproduced timeseries will be saved every 5000 learning epoch in "/training/example/trained_model/" and "/training/example/result_training/", respectively.
        
3-2. Test experiment (8 trials by one neural network may take about one and a half hours.)
        
        #Before running test trial, you have to copy “generate_00200000.npy” and “model_00200000.pth” (learning results) into "/test/example/"
        cp /training/example/result_training/generate_00200000.npy /test/example/
        cp /training/example/trained_model/model_00200000.pth /test/example/
        cd /test/example/
        python example_of_error_regression.py
        #Timeseries will be saved under “/test/example/result_ER/sit01/seq01 ~ seq08” for each trial
         
