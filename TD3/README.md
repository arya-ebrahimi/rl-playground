# Twin Delayed DDPG

TD3 can be seen as an improved version of DDPG, which utilizes clipped double q-learning, meaning that it learns two action value functions instead of one.

Also, the actor updates are delayed (updates are less frequent than the critic updates).

![](gifs/halfcheetah.gif) |  
:-------------------------:|
The result of trained DDPG agent after 500 episodes for HalfCheetah environment.

![](gifs/pendulum.gif) |  
:-------------------------:|
The result of trained DDPG agent after 500 episodes for Pendulum environment.