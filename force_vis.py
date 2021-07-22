import numpy as np
import matplotlib.pyplot as plt

class ForceVis():
    def __init__(self, online=False) -> None:
        self.data_label=['x','y','z','wx','wy','wz']
        self.data_color=['#F048A1','#F09D54','#603CF0','#28F024','#38D2F2','#183CF0']
        self.online = online
        self.data = []
        self.has_legend = False
        if online:
            plt.ion()
        pass

    def update(self, data):
        self.data.append(data)

    
    def show(self):
        t = range(len(self.data))
        vis_data = np.array(self.data)
        for i in range(6):
            plt.plot(t, vis_data[:, i], label=self.data_label[i], color=self.data_color[i])
        if self.online:
            plt.pause(1/240.)
        if not self.has_legend:
            plt.legend()
            self.has_legend = True
        plt.show()

