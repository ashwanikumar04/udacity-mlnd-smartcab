
# Machine Learning Engineer Nanodegree
## Reinforcement Learning
## Project: Train a Smartcab to Drive

Welcome to the fourth project of the Machine Learning Engineer Nanodegree! In this notebook, template code has already been provided for you to aid in your analysis of the *Smartcab* and your implemented learning algorithm. You will not need to modify the included code beyond what is requested. There will be questions that you must answer which relate to the project and the visualizations provided in the notebook. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide in `agent.py`.  

>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

-----

## Getting Started
In this project, you will work towards constructing an optimized Q-Learning driving agent that will navigate a *Smartcab* through its environment towards a goal. Since the *Smartcab* is expected to drive passengers from one location to another, the driving agent will be evaluated on two very important metrics: **Safety** and **Reliability**. A driving agent that gets the *Smartcab* to its destination while running red lights or narrowly avoiding accidents would be considered **unsafe**. Similarly, a driving agent that frequently fails to reach the destination in time would be considered **unreliable**. Maximizing the driving agent's **safety** and **reliability** would ensure that *Smartcabs* have a permanent place in the transportation industry.

**Safety** and **Reliability** are measured using a letter-grade system as follows:

| Grade 	| Safety 	| Reliability 	|
|:-----:	|:------:	|:-----------:	|
|   A+  	|  Agent commits no traffic violations,<br/>and always chooses the correct action. | Agent reaches the destination in time<br />for 100% of trips. |
|   A   	|  Agent commits few minor traffic violations,<br/>such as failing to move on a green light. | Agent reaches the destination on time<br />for at least 90% of trips. |
|   B   	| Agent commits frequent minor traffic violations,<br/>such as failing to move on a green light. | Agent reaches the destination on time<br />for at least 80% of trips. |
|   C   	|  Agent commits at least one major traffic violation,<br/> such as driving through a red light. | Agent reaches the destination on time<br />for at least 70% of trips. |
|   D   	| Agent causes at least one minor accident,<br/> such as turning left on green with oncoming traffic.       	| Agent reaches the destination on time<br />for at least 60% of trips. |
|   F   	|  Agent causes at least one major accident,<br />such as driving through a red light with cross-traffic.      	| Agent fails to reach the destination on time<br />for at least 60% of trips. |

To assist evaluating these important metrics, you will need to load visualization code that will be used later on in the project. Run the code cell below to import this code which is required for your analysis.


```python
# Import the visualization code
import visuals as vs

# Pretty display for notebooks
%matplotlib inline
```

### Understand the World
Before starting to work on implementing your driving agent, it's necessary to first understand the world (environment) which the *Smartcab* and driving agent work in. One of the major components to building a self-learning agent is understanding the characteristics about the agent, which includes how the agent operates. To begin, simply run the `agent.py` agent code exactly how it is -- no need to make any additions whatsoever. Let the resulting simulation run for some time to see the various working components. Note that in the visual simulation (if enabled), the **white vehicle** is the *Smartcab*.

### Question 1
In a few sentences, describe what you observe during the simulation when running the default `agent.py` agent code. Some things you could consider:
- *Does the Smartcab move at all during the simulation?*
- *What kind of rewards is the driving agent receiving?*
- *How does the light changing color affect the rewards?*  

**Hint:** From the `/smartcab/` top-level directory (where this notebook is located), run the command 
```bash
'python smartcab/agent.py'
```

**Answer:** 
The Udacity smartcab does not move from its position at which it start with in a trial. However, it moves to a random position in each trail. The agent recieves a positive reward when it waits on the red light and a negative reward when it does not move on the green signal and hinders the traffic.
            

### Understand the Code
In addition to understanding the world, it is also necessary to understand the code itself that governs how the world, simulation, and so on operate. Attempting to create a driving agent would be difficult without having at least explored the *"hidden"* devices that make everything work. In the `/smartcab/` top-level directory, there are two folders: `/logs/` (which will be used later) and `/smartcab/`. Open the `/smartcab/` folder and explore each Python file included, then answer the following question.

### Question 2
- *In the *`agent.py`* Python file, choose three flags that can be set and explain how they change the simulation.*
- *In the *`environment.py`* Python file, what Environment class function is called when an agent performs an action?*
- *In the *`simulator.py`* Python file, what is the difference between the *`'render_text()'`* function and the *`'render()'`* function?*
- *In the *`planner.py`* Python file, will the *`'next_waypoint()`* function consider the North-South or East-West direction first?*

**Answer:**
1. The flags which can be set are * **learning** *, * **update_delay** * and * **display** *
    - **learning**: This setting dictates if the agent should learn or not during the simulation
    - **update_delay**: This setting changes the time taken between the actions.
    - **display**: This allows us to run simulation with or without GUI. If set to ```true```, a ```PyGame``` GUI is launched which displays all the dummy agents and the Udacity car.
2. When the agent performs an action, ```act()``` method of ```environment.py``` is called which is evident from the below piece of the code
```self.env.act(self, action)```
3. ```render_text()``` is run to display the simulation progress on the command line while ```render()``` is run to display GUI when the simulation is run
4. In ```planner.py```, ```next_waypoint()``` considers ** East-West ** direction first which is evident from the below code

```
        # Next check if destination is cardinally East or West of location   
        
        elif dx != 0:

            if dx * heading[0] > 0:  # Heading the correct East or West direction
                return 'forward'
            elif dx * heading[0] < 0 and heading[0] < 0: # Heading West, destination East
                if dy > 0: # Destination also to the South
                    return 'left'
                else:
                    return 'right'
            elif dx * heading[0] < 0 and heading[0] > 0: # Heading East, destination West
                if dy < 0: # Destination also to the North
                    return 'left'
                else:
                    return 'right'
            elif dx * heading[1] > 0: # Heading North destination West; Heading South destination East
                return 'left'
            else:
                return 'right'

       # Finally, check if destination is cardinally North or South of location
       
        elif dy != 0:

            if dy * heading[1] > 0:  # Heading the correct North or South direction
              
```
                

-----
## Implement a Basic Driving Agent

The first step to creating an optimized Q-Learning driving agent is getting the agent to actually take valid actions. In this case, a valid action is one of `None`, (do nothing) `'left'` (turn left), `right'` (turn right), or `'forward'` (go forward). For your first implementation, navigate to the `'choose_action()'` agent function and make the driving agent randomly choose one of these actions. Note that you have access to several class variables that will help you write this functionality, such as `'self.learning'` and `'self.valid_actions'`. Once implemented, run the agent file and simulation briefly to confirm that your driving agent is taking a random action each time step.

### Basic Agent Simulation Results
To obtain results from the initial simulation, you will need to adjust following flags:
- `'enforce_deadline'` - Set this to `True` to force the driving agent to capture whether it reaches the destination in time.
- `'update_delay'` - Set this to a small value (such as `0.01`) to reduce the time between steps in each trial.
- `'log_metrics'` - Set this to `True` to log the simluation results as a `.csv` file in `/logs/`.
- `'n_test'` - Set this to `'10'` to perform 10 testing trials.

Optionally, you may disable to the visual simulation (which can make the trials go faster) by setting the `'display'` flag to `False`. Flags that have been set here should be returned to their default setting when debugging. It is important that you understand what each flag does and how it affects the simulation!

Once you have successfully completed the initial simulation (there should have been 20 training trials and 10 testing trials), run the code cell below to visualize the results. Note that log files are overwritten when identical simulations are run, so be careful with what log file is being loaded!
Run the agent.py file after setting the flags from projects/smartcab folder instead of projects/smartcab/smartcab.



```python
# Load the 'sim_no-learning' log file from the initial simulation results
vs.plot_trials('sim_no-learning.csv')
```


![png](output_11_0.png)


### Question 3
Using the visualization above that was produced from your initial simulation, provide an analysis and make several observations about the driving agent. Be sure that you are making at least one observation about each panel present in the visualization. Some things you could consider:
- *How frequently is the driving agent making bad decisions? How many of those bad decisions cause accidents?*
- *Given that the agent is driving randomly, does the rate of reliability make sense?*
- *What kind of rewards is the agent receiving for its actions? Do the rewards suggest it has been penalized heavily?*
- *As the number of trials increases, does the outcome of results change significantly?*
- *Would this Smartcab be considered safe and/or reliable for its passengers? Why or why not?*

**Answer:**
1. The driving agent is making bad decisions at an approximate rate of 42% in the starting which comes down to around 38% during the simulation and again reaches to around 40% in the end. Out of these bad decisions around 5% result in minor accidents and around 6% result in major accidents.
2. As the agent is choosing actions randomly, the reliability does not have any credibility which is resulting in constant delays varying from 85% to 75%.
3. The agent is constantly receiving rewards ranging from -5 in the beginning to -4.5 in the end. The rewards are being given to the agent on the random selection of the actions which in some way results in lessor major accidents in the end part of the simulation, however, this can't be considered as a relation between the safety and rewarding.
4. As the number of trials are increased, the outcome does not change drastically. The bad decisions do come down a little bit but accidents (both minor and major) remain almost same.
5. The agent in its current condition in no way be called ```SAFE```. The Smartcab with such high percentange of bad decisions which in turn results in the accidents to a threat to the passengers as well as other cars on the road.

-----
## Inform the Driving Agent
The second step to creating an optimized Q-learning driving agent is defining a set of states that the agent can occupy in the environment. Depending on the input, sensory data, and additional variables available to the driving agent, a set of states can be defined for the agent so that it can eventually *learn* what action it should take when occupying a state. The condition of `'if state then action'` for each state is called a **policy**, and is ultimately what the driving agent is expected to learn. Without defining states, the driving agent would never understand which action is most optimal -- or even what environmental variables and conditions it cares about!

### Identify States
Inspecting the `'build_state()'` agent function shows that the driving agent is given the following data from the environment:
- `'waypoint'`, which is the direction the *Smartcab* should drive leading to the destination, relative to the *Smartcab*'s heading.
- `'inputs'`, which is the sensor data from the *Smartcab*. It includes 
  - `'light'`, the color of the light.
  - `'left'`, the intended direction of travel for a vehicle to the *Smartcab*'s left. Returns `None` if no vehicle is present.
  - `'right'`, the intended direction of travel for a vehicle to the *Smartcab*'s right. Returns `None` if no vehicle is present.
  - `'oncoming'`, the intended direction of travel for a vehicle across the intersection from the *Smartcab*. Returns `None` if no vehicle is present.
- `'deadline'`, which is the number of actions remaining for the *Smartcab* to reach the destination before running out of time.

### Question 4
*Which features available to the agent are most relevant for learning both **safety** and **efficiency**? Why are these features appropriate for modeling the *Smartcab* in the environment? If you did not choose some features, why are those features* not *appropriate? Please note that whatever features you eventually choose for your agent's state, must be argued for here. That is: your code in agent.py should reflect the features chosen in this answer.
*

NOTE: You are not allowed to engineer new features for the smartcab. 

**Answer:**
Following features are relevant for learning:
- **waypoint**: This is required as the smartcab must know the direction it should drive to so that it can reach its destination.
- **light**: This is required as the smartcab will take decision based on the traffic signal to cross a juction or not.
- **left**: This input tells the intent of the vehicle on the left of the smartcab. It is important to know this as it may result in a major accident in case the traffic signal is red, the vehicle which is in the left of the smartcab is driving in forward direction and the agent decides to take a right turn.
- **oncoming**: This is important to know to avoid accidents when the smartcab takes a left turn on a junction.

Following features are not important:
- **deadline**: This tells the number of actions renaming in order to reach the destination which might have an impact on the efficiency but this can be ignored on a bigger picture. This can feed into the rewarding while the smartcab trains. With this ignored, ```Q``` table size is also reduced.
- ** right**: This input tells the intent of the vehicle on the right of the smartcab. On a junction, when the light is red, the smartcab should not go left or forward. Even, if it moves to the right, the lanes of both the cars will be different. Thus, knowing this input will not have much impact and ignoring it reduces ```Q``` table size too.


### Define a State Space
When defining a set of states that the agent can occupy, it is necessary to consider the *size* of the state space. That is to say, if you expect the driving agent to learn a **policy** for each state, you would need to have an optimal action for *every* state the agent can occupy. If the number of all possible states is very large, it might be the case that the driving agent never learns what to do in some states, which can lead to uninformed decisions. For example, consider a case where the following features are used to define the state of the *Smartcab*:

`('is_raining', 'is_foggy', 'is_red_light', 'turn_left', 'no_traffic', 'previous_turn_left', 'time_of_day')`.

How frequently would the agent occupy a state like `(False, True, True, True, False, False, '3AM')`? Without a near-infinite amount of time for training, it's doubtful the agent would ever learn the proper action!

### Question 5
*If a state is defined using the features you've selected from **Question 4**, what would be the size of the state space? Given what you know about the environment and how it is simulated, do you think the driving agent could learn a policy for each possible state within a reasonable number of training trials?*  
**Hint:** Consider the *combinations* of features to calculate the total number of states!

**Answer:**
In **Question 4**, I selected **waypoint**, **light**, **left** and **oncoming**.

- Waypoint: 3 states
- light: 2 states
- left: 4 states
- oncoming: 4 states

In total, ```3*2*4*4 = 96``` combinations need to calculated which a small number cosidering the agent in question is a smartcab which needs to take security seriously.






### Update the Driving Agent State
For your second implementation, navigate to the `'build_state()'` agent function. With the justification you've provided in **Question 4**, you will now set the `'state'` variable to a tuple of all the features necessary for Q-Learning. Confirm your driving agent is updating its state by running the agent file and simulation briefly and note whether the state is displaying. If the visual simulation is used, confirm that the updated state corresponds with what is seen in the simulation.

**Note:** Remember to reset simulation flags to their default setting when making this observation!

-----
## Implement a Q-Learning Driving Agent
The third step to creating an optimized Q-Learning agent is to begin implementing the functionality of Q-Learning itself. The concept of Q-Learning is fairly straightforward: For every state the agent visits, create an entry in the Q-table for all state-action pairs available. Then, when the agent encounters a state and performs an action, update the Q-value associated with that state-action pair based on the reward received and the iterative update rule implemented. Of course, additional benefits come from Q-Learning, such that we can have the agent choose the *best* action for each state based on the Q-values of each state-action pair possible. For this project, you will be implementing a *decaying,* $\epsilon$*-greedy* Q-learning algorithm with *no* discount factor. Follow the implementation instructions under each **TODO** in the agent functions.

Note that the agent attribute `self.Q` is a dictionary: This is how the Q-table will be formed. Each state will be a key of the `self.Q` dictionary, and each value will then be another dictionary that holds the *action* and *Q-value*. Here is an example:

```
{ 'state-1': { 
    'action-1' : Qvalue-1,
    'action-2' : Qvalue-2,
     ...
   },
  'state-2': {
    'action-1' : Qvalue-1,
     ...
   },
   ...
}
```

Furthermore, note that you are expected to use a *decaying* $\epsilon$ *(exploration) factor*. Hence, as the number of trials increases, $\epsilon$ should decrease towards 0. This is because the agent is expected to learn from its behavior and begin acting on its learned behavior. Additionally, The agent will be tested on what it has learned after $\epsilon$ has passed a certain threshold (the default threshold is 0.05). For the initial Q-Learning implementation, you will be implementing a linear decaying function for $\epsilon$.

### Q-Learning Simulation Results
To obtain results from the initial Q-Learning implementation, you will need to adjust the following flags and setup:
- `'enforce_deadline'` - Set this to `True` to force the driving agent to capture whether it reaches the destination in time.
- `'update_delay'` - Set this to a small value (such as `0.01`) to reduce the time between steps in each trial.
- `'log_metrics'` - Set this to `True` to log the simluation results as a `.csv` file and the Q-table as a `.txt` file in `/logs/`.
- `'n_test'` - Set this to `'10'` to perform 10 testing trials.
- `'learning'` - Set this to `'True'` to tell the driving agent to use your Q-Learning implementation.

In addition, use the following decay function for $\epsilon$:

$$ \epsilon_{t+1} = \epsilon_{t} - 0.05, \hspace{10px}\textrm{for trial number } t$$

If you have difficulty getting your implementation to work, try setting the `'verbose'` flag to `True` to help debug. Flags that have been set here should be returned to their default setting when debugging. It is important that you understand what each flag does and how it affects the simulation! 

Once you have successfully completed the initial Q-Learning simulation, run the code cell below to visualize the results. Note that log files are overwritten when identical simulations are run, so be careful with what log file is being loaded!


```python
# Load the 'sim_default-learning' file from the default Q-Learning simulation
vs.plot_trials('sim_default-learning.csv')
```


![png](output_24_0.png)



```python
# Load the 'sim_default-learning' file from the default Q-Learning simulation
vs.plot_trials('sim_default-learning.csv')
```


![png](output_25_0.png)


### Question 6
Using the visualization above that was produced from your default Q-Learning simulation, provide an analysis and make observations about the driving agent like in **Question 3**. Note that the simulation should have also produced the Q-table in a text file which can help you make observations about the agent's learning. Some additional things you could consider:  
- *Are there any observations that are similar between the basic driving agent and the default Q-Learning agent?*
- *Approximately how many training trials did the driving agent require before testing? Does that number make sense given the epsilon-tolerance?*
- *Is the decaying function you implemented for $\epsilon$ (the exploration factor) accurately represented in the parameters panel?*
- *As the number of training trials increased, did the number of bad actions decrease? Did the average reward increase?*
- *How does the safety and reliability rating compare to the initial driving agent?*

**Answer:**
- The observations are not similar in case of basic driving agent and the default Q-learning agent. Parameters like ```bad actions```, ```accidents``` and ```violations``` came down significantly in case of the default Q-learning agent. The reliability and rewards also increases with time in case of the Q-learning agent.
- It took 20 trails before testing. This does make sense as we are decreasing the epsilon in each trail by a factor of 0.05 and ```0.05*20```= 1.
- Yes, the implemented decaying function does represent the parameters panel.
- Yes, with the increase of the number of the trails, the bad actions do came down and the reward also increased.
- The overall number of bad decisions and resulted violations/accidented did came down at the end but the agent in its current condition no where fit for testing. That is why the safety and reliability grading is still ```F```.

-----
## Improve the Q-Learning Driving Agent
The third step to creating an optimized Q-Learning agent is to perform the optimization! Now that the Q-Learning algorithm is implemented and the driving agent is successfully learning, it's necessary to tune settings and adjust learning paramaters so the driving agent learns both **safety** and **efficiency**. Typically this step will require a lot of trial and error, as some settings will invariably make the learning worse. One thing to keep in mind is the act of learning itself and the time that this takes: In theory, we could allow the agent to learn for an incredibly long amount of time; however, another goal of Q-Learning is to *transition from experimenting with unlearned behavior to acting on learned behavior*. For example, always allowing the agent to perform a random action during training (if $\epsilon = 1$ and never decays) will certainly make it *learn*, but never let it *act*. When improving on your Q-Learning implementation, consider the implications it creates and whether it is logistically sensible to make a particular adjustment.

### Improved Q-Learning Simulation Results
To obtain results from the initial Q-Learning implementation, you will need to adjust the following flags and setup:
- `'enforce_deadline'` - Set this to `True` to force the driving agent to capture whether it reaches the destination in time.
- `'update_delay'` - Set this to a small value (such as `0.01`) to reduce the time between steps in each trial.
- `'log_metrics'` - Set this to `True` to log the simluation results as a `.csv` file and the Q-table as a `.txt` file in `/logs/`.
- `'learning'` - Set this to `'True'` to tell the driving agent to use your Q-Learning implementation.
- `'optimized'` - Set this to `'True'` to tell the driving agent you are performing an optimized version of the Q-Learning implementation.

Additional flags that can be adjusted as part of optimizing the Q-Learning agent:
- `'n_test'` - Set this to some positive number (previously 10) to perform that many testing trials.
- `'alpha'` - Set this to a real number between 0 - 1 to adjust the learning rate of the Q-Learning algorithm.
- `'epsilon'` - Set this to a real number between 0 - 1 to adjust the starting exploration factor of the Q-Learning algorithm.
- `'tolerance'` - set this to some small value larger than 0 (default was 0.05) to set the epsilon threshold for testing.

Furthermore, use a decaying function of your choice for $\epsilon$ (the exploration factor). Note that whichever function you use, it **must decay to **`'tolerance'`** at a reasonable rate**. The Q-Learning agent will not begin testing until this occurs. Some example decaying functions (for $t$, the number of trials):

$$ \epsilon = a^t, \textrm{for } 0 < a < 1 \hspace{50px}\epsilon = \frac{1}{t^2}\hspace{50px}\epsilon = e^{-at}, \textrm{for } 0 < a < 1 \hspace{50px} \epsilon = \cos(at), \textrm{for } 0 < a < 1$$
You may also use a decaying function for $\alpha$ (the learning rate) if you so choose, however this is typically less common. If you do so, be sure that it adheres to the inequality $0 \leq \alpha \leq 1$.

If you have difficulty getting your implementation to work, try setting the `'verbose'` flag to `True` to help debug. Flags that have been set here should be returned to their default setting when debugging. It is important that you understand what each flag does and how it affects the simulation! 

Once you have successfully completed the improved Q-Learning simulation, run the code cell below to visualize the results. Note that log files are overwritten when identical simulations are run, so be careful with what log file is being loaded!


```python
# Load the 'sim_improved-learning' file from the improved Q-Learning simulation
vs.plot_trials('sim_improved-learning.csv')
print("decay: 1.0/(self.t**2), tolerance: 0.05, alpha: 0.5" )
```


![png](output_30_0.png)


    decay: 1.0/(self.t**2), tolerance: 0.05, alpha: 0.5



```python
# Load the 'sim_improved-learning' file from the improved Q-Learning simulation
vs.plot_trials('sim_improved-learning.csv')
print("decay: 1.0/(self.t**2), tolerance: 0.01, alpha: 0.5" )
```


![png](output_31_0.png)


    decay: 1.0/(self.t**2), tolerance: 0.01, alpha: 0.5



```python
vs.plot_trials('sim_improved-learning.csv')
print("decay: 1.0/(self.t**2), tolerance: 0.01, alpha: 0.1" )
```


![png](output_32_0.png)


    decay: 1.0/(self.t**2), tolerance: 0.01, alpha: 0.1



```python
vs.plot_trials('sim_improved-learning.csv')
print("decay: (self.a)**self.t, tolerance: 0.01, alpha: 0.1, a: 0.1" )
```


![png](output_33_0.png)


    decay: (self.a)**self.t, tolerance: 0.01, alpha: 0.1, a: 0.1



```python
vs.plot_trials('sim_improved-learning.csv')
print("decay: (self.a)**self.t, tolerance: 0.01, alpha: 0.1, a: 0.01" )
```


![png](output_34_0.png)


    decay: (self.a)**self.t, tolerance: 0.01, alpha: 0.1, a: 0.01



```python
vs.plot_trials('sim_improved-learning.csv')
print("decay: (self.a)**self.t, tolerance: 0.01, alpha: 0.01, a: 0.01" )
```


![png](output_35_0.png)


    decay: (self.a)**self.t, tolerance: 0.01, alpha: 0.01, a: 0.01



```python
vs.plot_trials('sim_improved-learning.csv')
print("decay: math.exp(-(self.a * self.t)), tolerance: 0.01, alpha: 0.5, a: 0.1" )
```


![png](output_36_0.png)


    decay: math.exp(-(self.a * self.t)), tolerance: 0.01, alpha: 0.5, a: 0.1



```python
vs.plot_trials('sim_improved-learning.csv')
print("decay: math.exp(-(self.a * self.t)), tolerance: 0.01, alpha: 0.25, a: 0.1" )
```


![png](output_37_0.png)


    decay: math.exp(-(self.a * self.t)), tolerance: 0.01, alpha: 0.25, a: 0.1



```python
vs.plot_trials('sim_improved-learning.csv')
print("decay: math.fabs(math.cos(self.a*self.t)), tolerance: 0.01, alpha: 0.25, a: 0.1" )
```


![png](output_38_0.png)


    decay: math.fabs(math.cos(self.a*self.t)), tolerance: 0.01, alpha: 0.25, a: 0.1



```python
vs.plot_trials('sim_improved-learning.csv')
print("decay: math.fabs(math.cos(self.a*self.t)), tolerance: 0.01, alpha: 0.1, a: 0.1" )
```


![png](output_39_0.png)


    decay: math.fabs(math.cos(self.a*self.t)), tolerance: 0.01, alpha: 0.1, a: 0.1



```python
vs.plot_trials('sim_improved-learning.csv')
print("decay: math.exp(-(self.a * self.t)), tolerance: 0.01, alpha: 0.25, a: 0.1" )
```


![png](output_40_0.png)


    decay: math.exp(-(self.a * self.t)), tolerance: 0.01, alpha: 0.25, a: 0.1


### Question 7
Using the visualization above that was produced from your improved Q-Learning simulation, provide a final analysis and make observations about the improved driving agent like in **Question 6**. Questions you should answer:  
- *What decaying function was used for epsilon (the exploration factor)?*
- *Approximately how many training trials were needed for your agent before begining testing?*
- *What epsilon-tolerance and alpha (learning rate) did you use? Why did you use them?*
- *How much improvement was made with this Q-Learner when compared to the default Q-Learner from the previous section?*
- *Would you say that the Q-Learner results show that your driving agent successfully learned an appropriate policy?*
- *Are you satisfied with the safety and reliability ratings of the *Smartcab*?*

**Answer:**
##  Following functions were used

| #             | Epsilon Function      | # Trials|Tolerance|Alpha| a   |Safety|Reliability|
| ------------- |:---------------------:| -------:|----:|--------:|--:  |-----:|----------:|
| 1             |1/(t<sup>2</sup>)      | 20      |0.05  |0.5     |NA   |F     |F          |
| 2             |1/(t<sup>2</sup>)      | 20      |0.01  |0.1     |NA   |F     |F          |
| 3             |1/(t<sup>2</sup>)      | 20      |0.01  |0.1     |NA   |F     |D          |
| 4             |a<sup>t</sup>          | 20      |0.01  |0.1     |0.1  |D     |F          |
| 5             |a<sup>t</sup>          | 20      |0.01  |0.1     |0.01 |F     |F          |
| 6             |a<sup>t</sup>          | 20      |0.01  |0.01    |0.01 |F     |F          |
| 7             |e<sup>-at</sup>        | 20      |0.01  |0.5     |0.1  |A+    |A+         |
| 8             |e<sup>-at</sup>        | 20      |0.01  |0.25    |0.1  |A+    |A+         |
| 9             |cos(at)                | 20      |0.01  |0.25    |0.1  |A+    |A          |
| 10            |cos(at)                | 20      |0.01  |0.1     |0.1  |A+    |A          |
| 11            |e<sup>-at</sup>        | 100     |0.01  |0.25    |0.1  |A+    |A          |

- I tried multiple variations of the tolerance, alpha and epsilon function. The results related to bad actions, violations and accidents were not consistent with the basic Q-learning agent. The reliability increased significantly in case of the optimised agent. Also, the bad actions, violations and accidents almost came down to 0 for the optimised agent.

- Approximately 450 training trials were needed before testing with ```a=0.1```, ```alpha=0.25``` and ```tolerance=0.01```.

- The selected epsilon funtion is e<sup>-at</sup> with ```a=0.1```, ```alpha=0.25``` and ```tolerance=0.01``` which has 0 violations and accidents once the agent is trained. The reliability and reward also increased significantly. It learns from all the possible combination of the states.

- The optimised Q-learner is better in all terms than the default one. It has near to perfect stats in case of bad actions, violations and accidents. That is why it achieved **A+** rating for security and **A** for reliability i.e. to reach on time.

- Yes, the results suggest that the agent has learned an appropriate policy which is depicted in its ratings.

- Yes, I am satisfied with the ratings of the agent.





### Define an Optimal Policy

Sometimes, the answer to the important question *"what am I trying to get my agent to learn?"* only has a theoretical answer and cannot be concretely described. Here, however, you can concretely define what it is the agent is trying to learn, and that is the U.S. right-of-way traffic laws. Since these laws are known information, you can further define, for each state the *Smartcab* is occupying, the optimal action for the driving agent based on these laws. In that case, we call the set of optimal state-action pairs an **optimal policy**. Hence, unlike some theoretical answers, it is clear whether the agent is acting "incorrectly" not only by the reward (penalty) it receives, but also by pure observation. If the agent drives through a red light, we both see it receive a negative reward but also know that it is not the correct behavior. This can be used to your advantage for verifying whether the **policy** your driving agent has learned is the correct one, or if it is a **suboptimal policy**.

### Question 8

1. Please summarize what the optimal policy is for the smartcab in the given environment. What would be the best set of instructions possible given what we know about the environment? 
   _You can explain with words or a table, but you should thoroughly discuss the optimal policy._

2. Next, investigate the `'sim_improved-learning.txt'` text file to see the results of your improved Q-Learning algorithm. _For each state that has been recorded from the simulation, is the **policy** (the action with the highest value) correct for the given state? Are there any states where the policy is different than what would be expected from an optimal policy?_ 

3. Provide a few examples from your recorded Q-table which demonstrate that your smartcab learned the optimal policy. Explain why these entries demonstrate the optimal policy.

4. Try to find at least one entry where the smartcab did _not_ learn the optimal policy.  Discuss why your cab may have not learned the correct policy for the given state.

Be sure to document your `state` dictionary below, it should be easy for the reader to understand what each state represents.

**Answer:** 
I used following state marker

```waypoint + ":" + inputs['left'] + ":" + inputs['oncoming']+":" + inputs['light']```

An example optimal policy will be to wait on the red signal when the intent of the smartcab is to go forward with oncoming traffic.
The state-action will look like

| #  | State                   |Action |Policy    |
| -- |:-----------------------:|------:|------:   |
| 1  |forward:None:forward:red |None   |optimal   |
| 2  |forward:None:forward:red |left   |incorrect |
| 3  |forward:None:forward:red |right  |suboptimal|
| 4  |forward:None:forward:red |forward|incorrect |

An example for the same from the ```sim_improved-learning.txt``` at line#47.

```
forward:None:forward:red
 -- None : 2.11
 -- forward : -11.49
 -- right : 0.80
 -- left : -9.37

```

This result is because of reinforcement learning. The heighest value is for **None** which is correct in the optimal policy. The agent also learned a suboptimal policy **right** which is also correct it is allowed to take **right** when the signal is red.

One example of the sub-optimal policy can be as below (line#89)
```
right:right:forward:green
 -- None : -1.31
 -- forward : 0.62
 -- right : 0.25
 -- left : 0.00
```

As, can be seen here that even though ```waypoint``` is right, the agent will decide to go forward as it has the heighest value. This kind of sub-optimal policies one of the reasons for the reliability rating to be ```A``` instead of ```A+```.






-----
### Optional: Future Rewards - Discount Factor, `'gamma'`
Curiously, as part of the Q-Learning algorithm, you were asked to **not** use the discount factor, `'gamma'` in the implementation. Including future rewards in the algorithm is used to aid in propagating positive rewards backwards from a future state to the current state. Essentially, if the driving agent is given the option to make several actions to arrive at different states, including future rewards will bias the agent towards states that could provide even more rewards. An example of this would be the driving agent moving towards a goal: With all actions and rewards equal, moving towards the goal would theoretically yield better rewards if there is an additional reward for reaching the goal. However, even though in this project, the driving agent is trying to reach a destination in the allotted time, including future rewards will not benefit the agent. In fact, if the agent were given many trials to learn, it could negatively affect Q-values!

### Optional Question 9
*There are two characteristics about the project that invalidate the use of future rewards in the Q-Learning algorithm. One characteristic has to do with the *Smartcab* itself, and the other has to do with the environment. Can you figure out what they are and why future rewards won't work for this project?*

**Answer:**

> **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  
**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
