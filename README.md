# Reinforcement Learning in Finance

Repository for the course *Advanced Methods and its Applications - Reinforcement Learning in Finance* in the UIBK Data
Science Master's programme.

This course will cover the following contents

* Setting up a `gym` environment
* Teach a trading agent using Q-Learning
* Test and measure the performance of a trading agent
* Visualize results and compare them with other strategies
* Use Deep Reinforcement Learning to further optimize the trading agent

## Preparation and Setup

In order to solve the tasks following setup is required:

* Python == 3.8
* gym==0.21.0
* matplotlib==3.4.3
* notebook==6.4.5
* pandas==1.3.4
* cufflinks==0.17.3
* torch==1.10.0
* statsmodels==0.13.0
* `virtualenv` (optional)

### Setting up the virtual environment

This is an optional step to separate your local Python environment from packages required for this lab. If you don't
like to separate your environment you can simply skip this step.

In order to install the `virtualenv` package execute following command:

```
pip install virtualenv
```

The virtual environment can then be created via (make sure that you installed `python3.8`:

```
virtualenv -p python3.8 venv-rl-finance
```

Now the virtual environment can be activated by using following command:

Mac OS/Linux

```
source venv-rl-finance/bin/activate
```

Windows

```
venv-rl-finance\Scripts\activate
```

Your command line prompt should now start with `(venv-rl-finance)`. Any Python commands you use will now work with your
virtual environment.

In order to deactivate the virtual environment you can use the command:

```
deactivate
```

### Installing required packages

The required packages are listed in the `requirements.txt` file. In order to install all packages simply execute (if you
like to install it in the virtual environment you need to activate it previously):

```
pip install -r requirements.txt
```

#### M1 Chip problem

If you are using an Apple Computer with the new M1 chip you may face issues with installing scipy. To overcome these
issues please **miniconda** https://docs.conda.io/en/latest/miniconda.html. Each package can then be installed with

```
conda install somePackage
```

### Verify installed packages

After installing the required packages and activating the environment you can verify if everything works as expected by
running the following script

```
python package_verify.py
```

### Run jupyter notebook

If you use the virtual environment approach you need to specify register the kernel in the jupyter notebook:

```
python3.8 -m ipykernel install --user --name=venv-rl-finance
```

Now you should be able to run `jupyter notebook` via the following command:

```
jupyter notebook
```

Maybe you have to change the kernel to `venv-rl-finance`

This should open your default web browser and you will see the *Notebook Dashboard* which displays the content of your
current directory. Now you should be good to go :)

## Structure

The repository has the following structure

* `data`
    * `interest_rates_p1.csv`: List of timestamp + corresponding interest rate for product 1
    * `interest_rates_p2.csv`: List of timestamp + corresponding interest rate for product 2
    * `interest_rates_p3.csv`: List of timestamp + corresponding interest rate for product 3
* `q_learning`
    * `interest_rate_environment.py`: Interest rate `gym` environment
    * `interest_rate_q_learning.ipynb`: Jupyter notebook file for Q-Learning
* `deep_rl`
    * `data_analysis.py`: The analysis notebook showing a linear prediction model applied to the interest rates.
    * `interest_rate_environment_pytorch.py`: The adapted environment for training with Policy gradient algorithm by
      using Pytorch.
    * `policy_gradient.ipynb`: A notebook that implements the Policy gradient algorithm and applies it to the trading
      environment.
* `requirements.txt`: Contains all the required packages

## Assignment, due January 17th 2022

### Exercise 1: Q-Learning

The exercise consists of improving the discussed `interest_rate_environment.py` and applying Q-Learning with the
extended environment. The results should be discussed and evaluated.

#### Description

As the simple Q-Learning example shown in the seminar did not perform very well, the existing environment should be
extended to use a two dimensional observation space. In addition to the interest rate, the actual observation should
contain the current position as well. So an observation always is a tuple containing a value for the interest rate and
one for the current position. Possible observations are

* (interest_rate, SHORT)
* (interest_rate, NO_POSITION)
* (interest_rate, LONG)

Note that for the interest rates discrete values should be used as in the existing environment. The extension is
expected to yield better results, as the agent can now distinguish whether he already has an open position or not. Based
on this information he should be able to make better decisions.

#### Tasks

1. Create a `gym` environment which defines the observation space as a `Tuple(interest_rate, position)`. For
   the  `interest_rate` you should use discrete values as previously. Hint: The whole environment stays very similar as
   the existing one. Think about which parts need to be changed to use another observation space.

2. Apply Q-Learning by using the new environment. Again, this task is similar as showed in the seminar. Hint: Think
   about how the Q-Table changes. As you have now three different observations for each interest rate interval, also the
   size of Q-Table increases. Visualize and discuss the results (training & test) and compare it with the results using
   the simple environment. It would be nice if you find a way to print the Q-Table, even if it is substantially larger
   now.

3. Compare the results of the learned strategy using Q-Learning with the results using a very naive strategy. The naive
   strategy should open a LONG position if the current interest rate is positive, otherwise it should open a SHORT
   position. Implement the naive strategy and compare the result with the result using the learned RL strategy.

4. Play around with different values for the variables `alpha` (learning rate), `gamma` (discount_factor) and
   the `decay` in the learning process. How do they affect the learning process? Are there some other variables/learning
   parameters that are affecting the outcome substantially? What approach can be used to find "good" values for the
   mentioned variables? (Optional: Implement an algorithm that tries to find the optimal values.)

5. Discuss a few advantages and disadvantages of Reinforcement Learning for this usage scenario.

6. Do you think the performance can be increased by using the interest rates of multiple products for the training?
   Before doing the actual training, discuss in which cases such an approach could be helpful and when it could decrease
   the quality of the learned strategy? Train the strategy with multiple products and evaluate the learned strategy. Did
   the performance improve compared to using a single product for training?

### Exercise 2: Deep Reinforcement Learning

The deep reinforcement learning approach

### Tasks

1. In the lecture we saw how initializing the network with reasonable weights affects the performance of the agent. In
   this exercise you should train an agent by doing the same analysis steps with product 2. (`interest_rates_p2.csv`).
    - Analyze the auto-correlation of the interest rates
    - Analyze the partial-auto-correlation of the interest rates. Does it depend on more lags or fewer lags?
    - Calculate the init weights for product 2
    - Train a PG-Agent with the adapted parameters
    - Report the training-return and test return of the trained agent

**_Expected solution format:_**

Please hand in a single zip file which consists of

- Exercise 1:
    * `gym` environment
    * jupyter notebook file (file extension `.ipynb`) with Q-Learning, graphs and discussion
    * additional `.pdf`file if you prefer to discuss the results in a separate file
- Exercise 2:
    * Report `.ipynb` consisting of:
        * data analysis of product 2
        * trained agent for product 2
        * training and test return for agent

## Feedback & Support

We hope you enjoyed the course and you were able to learn something useful while completing the seminar and the
exercises. If you have feedback or encountered any issues, feel free to contact us at any time. Also, if you have any
difficulties in doing the exercises, we are here to help :)

We would really appreciate if you provide us some feedback using the following link:

[Link to Google Form](https://forms.gle/Dzxnj3dMXtv3QdS39)

Thanks!

**Contact:**

[Christoph Klösch](mailto:christoph.kloesch@student.uibk.ac.at)

[Markus Moosbrugger](mailto:markus.l.moosbrugger@student.uibk.ac.at)

 
