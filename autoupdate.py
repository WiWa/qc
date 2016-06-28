import numpy as np
import matplotlib.pyplot as plt
from time import sleep

# Demo of autoupdating a plot!

def getData():
    return np.random.random()

def update_plots(fig, ax, plots, xss, yss):
    if len(plots) is not len(xss) \
        or len(plots) is not len(yss) \
        or len(xss) is not len(yss):
        raise ValueError("plots, xss, and yss need to be the same length!")

    for i in range(len(plots)):
        plot = plots[i]
        xs = xss[i]
        ys = yss[i]
        plot.set_xdata(xs)
        plot.set_ydata(ys)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    plt.pause(0.0001)
    # print("wtf")

plt.ion()

np.random.seed()

ts = np.linspace(0, 42, 42)

partial_ts = []
partial_data1 = []
partial_data2 = []

fig, ax = plt.subplots()
p1, = plt.plot(partial_ts, partial_data1)
p2, = plt.plot(partial_ts, partial_data2)
plt.show()
plt.pause(0.0001)

for t in ts:
    data1 = getData()
    data2 = getData()
    partial_ts.append(t)
    partial_data1.append(data1)
    partial_data2.append(data2)
    print((data1, data2))
    update_plots(fig, ax, \
        [p1, p2], \
        [partial_ts, partial_ts], \
        [partial_data1, partial_data2])
    sleep(0.1)
print("Done! Press Enter to end.")
raw_input()
