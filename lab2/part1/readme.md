# CS5340 Assignment 2 Part 1: Junction Tree

## Usage

### Data

The structure of the assignment 2 part 1 directory should be as follows:
```bash
[part1]
-- brute_force.py  # solution script that computes brute force result (joint)
-- main.py  # run this
-- jt_construction.py  # contains helper functions to construct the junction tree
-- factor_utils.py  # some factor utilities such as marginalization, product, reduction.
-- factor.py  # factor class definition.
-- readme.md
-- data/
    -- inputs/  # input files for each case
        -- 1.json
        -- 2.json 
        ...
    -- predictions/  # prediction files (your results) for each case
        -- 1.json
        -- 2.json 
        ...
    -- answers/  # answer files (our results) for each case
        -- 1.json
        -- 2.json 
        ...
```


#### Input File Format

The input `.json` files are encoded in the following format:
```bash
{
    "nodes": ["1", "2", ..., "n"],  # list of N nodes
    
    "edges": [["1","2"], ..., ["2", "n"]]  # E x 2 edge list

    "factors": {
        {
          "var": [0,2],  # scope of factor
          "card": [2, 2],  # cardinality of each variable
          "val": [1, 2, 3, 4]  # potential for each combination of variable realization.
        },
        ...
    }

    "evidence": {
        "2": 0  # implies that 2 is observed to be 1.    
    }
}
```

#### Prediction/Answer File Format

The prediction/answer `.json` files are encoded in the following format:
```bash
{
    "0": [0.3333, 0.6666]  # p(X0 = 0) = 0.3333; p(X0=1) = 0.6666
    ...
}
```
Note that the cardinality of variables can be more than 2.

### Making predictions:

To make predictions for e.g. `inputs/1.json`:
```bash
python main.py --case 1
```
which will create the prediction file `inputs/1.json`. For correctness, the predicted values should be close to the 
answer files that we have provided (results might differ based on the operators used). 

We will evaluate your score for this task based on code and results (small numerical differences will not be much of an 
issue as long as your code is corect).

Alternatively, to run for all test cases, you can run `predict_all_cases.sh`.

### Questions:

If there are issues, please post them in the CS5340 forum. We want to ensure that students have access to the 
same amount of help through the forums.
