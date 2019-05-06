
In this quick notebook, we will be dicussing Bayesian Statisitcs over Bayesian Networks and Inferencing them using Pgmpy Python library.
Bayesian statistics is a theory in the field of statistics based on the Bayesian interpretation of probability where probability expresses a degree of belief in an event, which can change as new information is gathered, rather than a fixed value based upon frequency or propensity.Bayesian statistical methods use Bayes' theorem to compute and update probabilities after obtaining new data. Bayes' theorem describes the conditional probability of an event based on data as well as prior information or beliefs about the event or conditions related to the event.





# Bayes' theorem
Bayes' theorem is a fundamental theorem in Bayesian statistics, as it is used by Bayesian methods to update probabilities, which are degrees of belief, after obtaining new data. Given two events A and B, the conditional probability of A given that B is true is expressed as follows:
![Bayes Theorem](https://github.com/anmolkapoor/inference-bayesian-networks-using-pgmpy-example-social-media-moderator/raw/master/images/bayes_theorem.png)

The probability of the evidence P(B) can be calculated using the law of total probability. If ![A values](https://github.com/anmolkapoor/inference-bayesian-networks-using-pgmpy-example-social-media-moderator/raw/master/images/A_values.png) is a partition of the sample space, which is the set of all outcomes of an experiment, then:
![total probs](https://github.com/anmolkapoor/inference-bayesian-networks-using-pgmpy-example-social-media-moderator/raw/master/images/total_probability.png)



# Bayesian network
A Bayesian network is a probabilistic graphical model (a type of statistical model) that represents a set of variables and their conditional dependencies via a directed acyclic graph (DAG). Bayesian networks are ideal for taking an event that occurred and predicting the likelihood that any one of several possible known causes was the contributing factor. 




# Example
### Statistical moderator for social platform with a given information such as user history, ML model prediction, other user flagging the content, etc
We can use bayes rule and total probability theorem to infer probabilites in a bayes network. Lets take an example:

Example
Lets consider an example, where a social media website wish to moderate content on the site and suspends bad user accounts. For this they would like us to create a statistical moderator that can take the preemtive measure based on information given. Lets assume we have following information:
* M : A prediction from a ML model that can read the content and give a score (probability) that this content should be flagged.
* U :  Another User flags the content.
* B : The account was suspended before for any bad content.
* R : Score (Probability) that the content should be removed from the platform.
* S : Score (Probability) that account should be suspended

Lets assume probabilities are given to us for the network as follows:
![Network](https://github.com/anmolkapoor/inference-bayesian-networks-using-pgmpy-example-social-media-moderator/raw/master/images/bayes_net_final.jpg)
![Probs](https://github.com/anmolkapoor/inference-bayesian-networks-using-pgmpy-example-social-media-moderator/raw/master/images/probs.png)
<image prob>



**Lets create this bayes network in python using pgmpy library https://github.com/pgmpy/pgmpy .**


```python
!pip install pgmpy
```

    Requirement already satisfied: pgmpy in /anaconda3/lib/python3.7/site-packages (0.1.7)
    Requirement already satisfied: scipy>=1.0.0 in /anaconda3/lib/python3.7/site-packages (from pgmpy) (1.2.1)
    Requirement already satisfied: networkx<1.12,>=1.11 in /anaconda3/lib/python3.7/site-packages (from pgmpy) (1.11)
    Requirement already satisfied: numpy>=1.14.0 in /anaconda3/lib/python3.7/site-packages (from pgmpy) (1.16.2)
    Requirement already satisfied: decorator>=3.4.0 in /anaconda3/lib/python3.7/site-packages (from networkx<1.12,>=1.11->pgmpy) (4.4.0)



```python
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np
```


```python
bayesNet = BayesianModel()
bayesNet.add_node("M")
bayesNet.add_node("U")
bayesNet.add_node("R")
bayesNet.add_node("B")
bayesNet.add_node("S")

bayesNet.add_edge("M", "R")
bayesNet.add_edge("U", "R")
bayesNet.add_edge("B", "R")
bayesNet.add_edge("B", "S")
bayesNet.add_edge("R", "S")
```

Adding CPDs for each node. A quick note is that while adding proabilities, we have to give FALSE values first.


```python
cpd_A = TabularCPD('M', 2, values=[[.95], [.05]])
cpd_U = TabularCPD('U', 2, values=[[.85], [.15]])
cpd_H = TabularCPD('B', 2, values=[[.90], [.10]])

cpd_S = TabularCPD('S', 2, values=[[0.98, .88, .95, .6], [.02, .12, .05, .40]],
                   evidence=['R', 'B'], evidence_card=[2, 2])

cpd_R = TabularCPD('R', 2,
                   values=[[0.96, .86, .94, .82, .24, .15, .10, .05], [.04, .14, .06, .18, .76, .85, .90, .95]],
                   evidence=['M', 'B', 'U'], evidence_card=[2, 2,2])
bayesNet.add_cpds(cpd_A, cpd_U, cpd_H, cpd_S, cpd_R)
```

Checking if model is correctly added.


```python
bayesNet.check_model()
print("Model is correct.")
```

    Model is correct.


Creating solver that uses variable elimination internally for inference.


```python
solver = VariableElimination(bayesNet)
```

Lets take some examples. For cross verification, we will be doing inference manually also using Bayes Theorem and Total Probability theorem.

#### 1. Lets find proability of "Content should be removed from the platform"**

```
P(R) 
=P(R|MBU)*P(M)*P(B)*P(U)+P(R|MBU)*P(M)*P(B)*P(!U)+P(R|MBU)*P(M)*P(!B)*P(U)
+P(R|MBU)*P(M)*P(!B)*P(!U)+P(R|MBU)*P(!M)*P(B)*P(U)+P(R|MBU)*P(!M)*P(B)*P(!U)
+P(R|MBU)*P(!M)*P(!B)*P(U)+P(R|MBU)*P(!M)*P(!B)*P(!U) --- [Using total probability theorem as R depends on M, B, U]
=0.95*0.05*0.1*0.15+0.9*0.05*0.1*0.85+0.85*0.05*0.9*0.15
+0.76*0.05*0.9*0.85+0.18*0.95*0.1*0.15+0.06*0.95*0.1*0.85
+0.14*0.95*0.9*0.15+0.04*0.95*0.9*0.85
=0.09378
```

Using pgmpy library:


```python
result = solver.query(variables=['R'])
print("R", result['R'].values[1])
```

    R 0.09378000000000002


#### 2. Lets find probability of "Content should be removed from platform given our ML model flags it"

````
P(R|A) 
= P(R|AHU) * P(H) * P(U) + P (R|AH!U) * P(H) *P(!U) + P(R|A!HU) 
* P(!H) * P(U)+ P(R|A!H!U) * P(!H) * P(!U)                      -------- [ Using Total Probability theorem ]
=0.95*0.1*0.15 + 0.9*0.1*0.85 +0.85*0.9*0.15 + 0.76*0.9*0.85
=0.7869
````

Now, Using pgmpy libary:


```python
result = solver.query(variables=['R'], evidence={'M': 1})
print("R| M", result['R'].values[1])

```

    R| M 0.7869


#### Pgmpy can also find complex proability inference considering dependent and independent variable considering something is given.
#### For example, we can find "Account should be suspended given it was suspened before"


```python
result = solver.query(variables=['S'], evidence={'B': 1})
print("S| B", result['S'].values[1])
```

    S| B 0.15345299999999998


#### Model has other features such as it can also find dependencies between the variables. Example:


```python
bayesNet.get_independencies()
```




    (M _|_ U, B)
    (M _|_ B | U)
    (M _|_ U | B)
    (M _|_ S | R, B)
    (M _|_ S | R, U, B)
    (U _|_ M, B)
    (U _|_ B | M)
    (U _|_ M | B)
    (U _|_ S | R, B)
    (U _|_ S | M, R, B)
    (B _|_ M, U)
    (B _|_ U | M)
    (B _|_ M | U)
    (S _|_ M, U | R, B)
    (S _|_ U | M, R, B)
    (S _|_ M | R, U, B)




```python
print("Completed.")

```

    Completed.

