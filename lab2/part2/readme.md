# CS5340 Assignment 2 Part 2: Parameter Learning

## Usage

### Data

The structure of the assignment 2 part 2 directory should be as follows:
```bash
[part2]
-- main.py
-- readme.md
-- data/
    -- observations/  # observation files for each case
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


#### Observation File Format

The observation `.json` files are encoded in the following format:
```bash
{
    "nodes": ["1", "2", ..., "n"],  # list of N nodes
    
    "edges": [["1","2"], ..., ["2", "n"]]  # E x 2 edge list
    
    "nobservations": O  # number of observations

    "observations": {
        "1": [0.3, ..., 0.5]  # list of O observations for node "1"
        ...
        "n": [0.566, ..., 0.51234]  # list of O observations for node "n"
    }
}
```

#### Prediction/Answer File Format

The prediction/answer `.json` files are encoded in the following format:
```bash
{
    "1": {
        "bias": 0.1005  # w0 value for node 1
        "variance": 1.0527  # sigma value for node 1
    }

    "2": {
        "bias": 0.3358  # w0 value for node 2
        "variance": 0.0959  # sigma value for node 2,
        "1": 0.4775  # w1 value to weight observation of node 1, which is a  parent of node 2.
    }
    
    ...
    
    "n": {
        ...
    }
}
```

### Making predictions:

To make predictions for e.g. `observations/1.json`:
```bash
python main.py --case 1
```
which will create the prediction file `predictions/1.json`. For correctness, the predicted values should be close to the 
answer files that we have provided (results might differ based on the operators used). 

We will evaluate your score for this task based on code and results (small numerical differences will not be an issue).

### Questions:

If there are issues, please post them in the CS5340 forum. We want to ensure that students have access to the 
same amount of help through the forums.
