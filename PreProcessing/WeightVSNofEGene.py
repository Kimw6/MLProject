from matplotlib import pyplot as plt

elist = []
with open('../dataOther/elist.txt', 'r') as f:
    for line in f:
        elist.append(int(line))

print(elist)
wlist = []
for ww in range(1, 1000, 10):
    wlist.append(ww)
wlist.append(999)
print(len(wlist))
print(len(elist))


plt.plot(wlist, elist)
plt.xlabel('Weight range')
# naming the y axis
plt.ylabel('Number of essential genes')
plt.title('Number of essential genes by weight range')
plt.ylim(ymin=0)
plt.ylim(ymax=250)
plt.grid(True)
plt.show()