# Reward_eee_legged_robot
All the credit belong to https://github.com/leggedrobotics/legged_gym/tree/master, legged_gym github creators

1. How to change terrain?
2. landing stability reward ? 增加了
    參數
    legged_robot_config.py :
        class rewards :
            class scales :
                landing_stability: 1.0 # 著地穩定性
            landing_force_threshold: 50.0 # 著地穩定性
    邏輯
    legged_robot.py :
        def _reward_landing_stability(self):