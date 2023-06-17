import math
import random
import matplotlib.pyplot as plt
import numpy

#arm 0 is assigned to be optimal

print("give number of arms: ")
k = int(input())



expected_reward = [0.0]
emp_regret = 0
cumm_regret = 0

A = set(range(k))
comm_reward = [0.0 for a in A] 


print("give value of DELTA in range (0,1) preferable small: ")
DELTA = float(input())


print("give number of iterations")
tot_iter = int(input())

arm_count = [0 for a in A]

emp_reg = []
com_reg =[]

def sample_reward(arm):

    mean = 0.5
    std_dev = 0.1

    p = 0.5

    if arm != 0:
      #sample = numpy.random.normal(mean, std_dev)
      #sample = max(0, min(sample, 1))
      sample  = numpy.random.binomial(1,p)
      return sample

    else :
      #sample = numpy.random.normal(mean+DELTA, std_dev)
      #sample = max(0, min(sample, 1))
      sample  = numpy.random.binomial(1,p+DELTA)
      return sample


def epsilon_t(t,k):
    first = 1.0/k
    if(t == 0):
        return 1.0/k
    temp = k*t
    second = math.sqrt(math.log(k) * 1.0 / temp)
    ans = min(first,second)

    return min(first,second)





def prob(t,k,a):
    first = (1-k*epsilon_t(t,k))*1.0
    numer = 1.0*math.exp(  epsilon_t(t-1,k)* comm_reward[a]  )
  
    denom = 0
    for i in range(0,len(A)):
        denom += 1.0*math.exp(  epsilon_t(t-1,k)* comm_reward[i]  )

    return (first*numer/denom)+epsilon_t(t,k)

def emp_comm_regret(comm_reward,arm):

    oracle_new = comm_reward[0]
    emp_reg = oracle_new - comm_reward[arm]

    
    return emp_reg


oneTimeOnly = 0


for i in range(1,tot_iter,1):
    
    print("_________________________________________________")
    print(i)

    current_prob = [prob(i,k,a) for a in A]

    temp = set(current_prob)
    Action_chosen = random.choices(list(A), weights=current_prob)[0]


    arm_count[Action_chosen] += 1

    result = all(element == current_prob[0] for element in current_prob)
    
    if(result):
        arm = random.randint(0,len(current_prob)-1)
        print("bad")
    else:
        arm = Action_chosen
            
        
    reward = sample_reward(arm)

    emp_regret += emp_comm_regret(comm_reward,arm)

    ## check it 

    if arm != 0:
      cumm_regret += DELTA ## check this 


    com_reg.append(cumm_regret)

    emp_reg.append(emp_regret)


    comm_reward[arm] = comm_reward[arm]+ reward/current_prob[arm]
   
    #print("iteration NO: ",i)
    #print("arm: ",arm)
    #print("reward: ",reward)
    #print("probabilty: ",current_prob)
    #print("commulative regret: ",cumm_regret)


    print("_________________________________________________")


plt.plot(com_reg)
plt.show()
print("how many times a arm is selected: ",arm_count)