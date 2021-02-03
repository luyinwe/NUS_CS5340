# CS5340: Monte Carlo Inference Assignment

In `part1/main.py` and `part2/main.py`, you will implement Importance sampling and Gibbs sampling respectively. There 
are four cases for each part i.e. `1, 2, 3, 4`:

To execute your scripts, you may run:
```
cd [ROOT]/part1
python main.py --case <CASE-NUM>  # e.g. python main.py --case 1

cd [ROOT]/part2
python main.py --case <CASE-NUM>  # e.g. python main.py --case 1
```

This will save your results into `part1/data/predictions` and `part2/data/predictions` respectively. You may compare 
the ground-truth using `check_answers.py` e.g.
```
cd [ROOT]/part1
python check_answers.py --case <CASE-NUM>  # e.g. python main.py --case 1

cd [ROOT]/part2
python check_answers.py --case <CASE-NUM>  # e.g. python main.py --case 1
```
We will use a tolerance of up to 1 decimal place w.r.t. to teh ground-truth.
