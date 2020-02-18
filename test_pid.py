import numpy as np
import matplotlib.pyplot as plt

class Oven:
    def __init__(self, outside_temp=70, temp=100):
        self.temp = temp
        self.outside_temp = outside_temp

    def update_temp(self, heat_level):
        # Per time_step, it will start with decreasing by 0.3
        diff = self.temp-self.outside_temp
        cool_temp = diff*0.01
        if heat_level>= 5:
            heat_level = 5
        temp = self.temp - cool_temp + heat_level
        self.temp = temp
        return temp

class PID_controller:
    def __init__(self, kp, ki, kd, goal_temp, bias=4, time_step=1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.bias = bias
        self.prev_temp = 100
        self.goal_temp = goal_temp
        self.prev_error = goal_temp-self.prev_temp
        self.time_step = time_step
        self.i_total = 0

    def update(self, new_temp):
        error = self.goal_temp - new_temp
        p_term = error*self.kp + self.bias
        self.i_total += error*self.time_step
        if self.i_total > 50:
            self.i_total = 50
        elif self.i_total < -50:
            self.i_total = -50
        i_term = self.i_total*self.ki
        d_term = (self.prev_error - error)/self.time_step*self.kd
        return p_term + i_term + d_term

oven = Oven()
pid_controller = PID_controller(kp=0.7, ki=0.01, kd=0.001, goal_temp=500, bias=4)
all_temp = []
temp = 100
for i in range(2000):
    u = pid_controller.update(temp)
    temp = oven.update_temp(u)
    all_temp.append(temp)

plt.plot(range(len(all_temp)), all_temp)
plt.title("P controller")
plt.show()
