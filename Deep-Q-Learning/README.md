### DQN on gym Taxi environment
![](https://github.com/Arya-Ebrahimi/RL-Playground/blob/main/Deep-Q-Learning/runs/out.gif)
#### Final Results
|![](https://github.com/Arya-Ebrahimi/RL-Playground/blob/main/Deep-Q-Learning/images/50000.png)|
|:--:|
| <b>Image Credits - Fig.2 - 4K Mountains Wallpaper</b>|
##### My first experiment was a DQN similar to the one proposed in Pytorch, but it didn't learn anything.
![](https://github.com/Arya-Ebrahimi//RL-Playground/blob/main/Deep-Q-Learning/images/target_network_soft_network.png?raw=true)

##### In my next experiments, Instead of using the soft update of the target network's weights, I used the method of replacing the target weights with the policy network weights. It got better but still was very noisy. I didn't test this version with a larger number of episodes. Instead, I reduced the maximum timesteps for each episode. The default maximum length of an episode is 200 for the Taxi environment, but I reduced it to 100, and the learning process improved.

![](https://github.com/Arya-Ebrahimi//RL-Playground/blob/main/Deep-Q-Learning/images/replace_target_with_policy.png?raw=true)
![](https://github.com/Arya-Ebrahimi//RL-Playground/blob/main/Deep-Q-Learning/images/reduce_maximum_timestep.png?raw=true)
