
Synthetic categorical XAI datasets
---------------------------------------
The datasets are part of the paper "Evaluation of post-hoc XAI approaches through synthetic tabular data"
Please refer to http://www.dmir.uni-wuerzburg.de/projects/deepscan/xai-eval-data/ for paper references and citation.


Data Structure
---------------------------------------
The datasets contain 12 Boolean features with every possible permutation being included exactly once in the dataset,
resulting in 2*12=4096 data samples in each dataset.
The labels 'y' are generated as described in the paper, with the first columns being used for label calculation.
(i.e. for the XOR dataset the label is calculated by y = f1 XOR f2)
The 'explanation' column contains the relevant feature columns for each data sample,
according to the definition given in the paper.


Dataset Use
---------------------------------------
The dataset is constructed to allow quick processing in python, using the pandas library.

Dataset loading example:
    import ast
    import pandas as pd
    dataset = pd.read_csv('../datasets/boolean/XOR.csv')
    y = dataset['y']
    #  Parse explanation String to python set, so relevant features in expl
    #  can then be compared to the feature column names
    expl = dataset['explanation'].apply(lambda x: set(ast.literal_eval(x)))
    X = dataset.drop(['y', 'explanation'], axis=1)
