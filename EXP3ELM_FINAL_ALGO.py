import math
import random
import numpy
import matplotlib.pyplot as plt
import time

startTime = time.time()

# number of arms are being chosen randomly we can give K and DELTA explicitely
# K = random.randint(2, 10) # to decide no of arms
print("give number of arms: ")
K = int(input())

print("give value of DELTA in range (0,1) preferable small: ")
DELTA = float(input())


print("give value of confidence parameter in range (0,1)")
delta = float(input())


print("give number of iterations")
tot_iter = int(input())


# arm 0 is assigned to be the optimal arm beforehand

def sample_reward(arm, DELTA):
    mean = 0.5
    std_dev = 0.1

    p = 0.5

    if arm != 0:
        # sample = numpy.random.normal(mean, std_dev)
        # sample = max(0, min(sample, 1))
        sample = numpy.random.binomial(1, p)
        return sample

    else:

        ## value of delta can be changed from here
        # sample = numpy.random.normal(mean+DELTA, std_dev)
        # sample = max(0, min(sample, 1))
        sample = numpy.random.binomial(1, p + DELTA)
        return sample


def emp_comm_regret(comm_reward, arm):
    oracle_rew = comm_reward[0]
    emp_reg = oracle_rew - comm_reward[arm]

    return emp_reg


def decide_arm(K, reg, com_reg,arm_freq):
    # intialize the arm set
    A = set(range(K))
    print(A)
    A0 = A

    rHat = {a: 0 for a in A}
    V_R = {a: 0 for a in A}

    emp_regret = 0  # cummilative emperical regret
    cumm_regret = 0  # cummilative expected regret
    comm_reward = [0.0 for a in A]  # cummilative rHat for each action

    print(A)
    B = 4 * (math.exp(1) - 2) * (2 * math.log(K) + math.log(2.0 / delta))

    t = 0
    i = 1
    epsilon_0 = 1 / K
    epsilon_t = -1
    rho_tilde = {}

    while True:

        t += 1
        rho_tilde = {}
        print("___________________________")
        print(t)

        if i == 1:
            epsilon_t_minus_1 = epsilon_0
            epsilon_t = min(1 / K, math.sqrt(math.log(K) / (K * t)))

            i = -1
        else:
            epsilon_t_minus_1 = epsilon_t
            epsilon_t = min(1 / K, math.sqrt(math.log(K) / (K * t)))

        # calculating probability for each arm in runtime
        denom = sum([math.exp(epsilon_t_minus_1 * rHat[a_prime]) for a_prime in A])
        for a in A:
            numer = (1 - len(A) * epsilon_t) * math.exp(epsilon_t_minus_1 * rHat[a])

            prob = ((numer / denom) + epsilon_t)
            rho_tilde[a] = prob

        # will return random
        Action_chosen = random.choices(list(A), weights=list(rho_tilde.values()))[0]

        arm_freq[Action_chosen] += 1

        print("Action_chosen:  ", Action_chosen)

        # have to change this thing
        Rt = sample_reward(Action_chosen, DELTA)
        reward = Rt

        emp_regret += emp_comm_regret(comm_reward, Action_chosen)

        if Action_chosen != 0:
            cumm_regret += DELTA

        com_reg.append(cumm_regret)
        reg.append(emp_regret)

        comm_reward[Action_chosen] = comm_reward[Action_chosen] + reward / rho_tilde[Action_chosen]

        At_star_max = -1
        At_star = -1

        for a in A:
            if a == Action_chosen:
                rHat[a] += Rt / rho_tilde[a]
            else:
                rHat[a] += 0

            if At_star_max < rHat[a]:
                At_star_max = rHat[a]
                At_star = a

            V_R[a] += 1 / rho_tilde[a]

        print("rHat: ", rHat)

        print("At_star: ", At_star, "At_star_max: ", At_star_max)

        arms = []
        for a in A:

            LHS = At_star_max - rHat[a]
            RHS = math.sqrt(B * (V_R[At_star] + V_R[a]))

            if LHS <= RHS:
                arms.append(a)

        print("arms after exclusion: ", arms)
        A = arms

        print("________________________________________")

        chosen_arm = list(A)[0]

        if len(A) == 1:

            break
        elif t == tot_iter:
            break

    return chosen_arm


reg = []  # sequence of emperical regret for plotting
com_reg = []  # sequence of expected regret for plotting
arm_freq = [0] * K

choose_arm = decide_arm(K, reg, com_reg,arm_freq)
print(f"The chosen arm is {choose_arm}.")

plt.plot(com_reg)
plt.xlabel("iterations")
plt.ylabel("imp_regret")
plt.show()

print("arm frequencies ", arm_freq)

executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))