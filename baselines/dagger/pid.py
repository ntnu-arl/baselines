import numpy as np
    
class PID:
    def __init__(self, Kp, Kd):
        self.Kp = np.array(list(Kp))
        self.Kp = np.diag(self.Kp)
        
        self.Kd = np.array(list(Kd))
        self.Kd = np.diag(self.Kd)

        print('Kp:', self.Kp)
        print('Kd:', self.Kd)

    def calculate(self, obs):
        ctrl = self.Kp.dot(obs[0:3]) + self.Kd.dot(obs[3:6])
        #print('obs:', obs)
        #print('ctrl:', ctrl)
        return ctrl