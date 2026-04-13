# Final Report

## Project Outcome
This report summarizes the final algorithm runs and student-pair assignments.

## Student Pair Assignment
- Pratyush Kumar and Rishabh worked on PPO.
- Shantanu Dukare worked on A2C.
- Rakshit Verma and Rahul Soni worked on A3C.

## Final Results

| Algorithm | Train Time (s) | Timesteps | Mean Reward | Std Reward | Mean Score | Eval Episodes | Model Path                                                             |
| --------- | -------------: | --------: | ----------: | ---------: | ---------: | ------------: | ---------------------------------------------------------------------- |
| PPO       |        645.893 |    100000 |     909.450 |    459.906 |     63.700 |            10 | results/ppo_only/ppo_model.pt                                          |
| A2C       |        498.605 |    100000 |      12.960 |      6.507 |      0.300 |            10 | results/a2c_only/a2c_model.pt                                          |
| A3C       |         16.028 |    100003 |    1291.000 |      0.000 |     91.000 |            10 | /Users/rakshitverma/Documents/flappyBird/results/a3c_only/a3c_model.pt |


## Result Sources
- [A3C JSON](results/a3c_only/a3c_result.json)
- [PPO JSON](results/ppo_only/ppo_result.json)
- [A2C JSON](results/a2c_only/a2c_result.json)
