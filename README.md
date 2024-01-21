# StratifiedGraphIndexing
```sg_main.py``` is a script for indexing and searching in high-dimensional deep descriptor databases. ```sg.py``` and ```sg_indexing.py``` contains the algorithms for indexing and searching. Here,the results are evaluated based on Precision,Recall,F1-score and Searching time. Precision,Recall,F1-score are measured against sklearn.neighbors' brute search method.

## Installation
* numpy
* sklearn
* ml_metrics
* time



## Indexing and Searching

Make the following changes to ```sg_main.py```

Step1:  Load the data in data_path.  
Step2:  Change the parameters M and ef  
Step3:  Change the query indexes (e.g. [0,15,123,500,888,1234,3456,4567,7890,8888]) to search  
Step4:  Run ``` python sg_main.py ```  
Step5:  Input the number of nearest neighbors to return (e.g. 100)  
Step6:  For the first time and indexes will be created and saved inside the folder.  
Step7:  Repeat Step3 to Step5 to get nearest neighbors for different features and different depth.
