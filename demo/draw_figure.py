import matplotlib.pyplot as plt
import numpy as np
import pdb


X = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 2.5,3.0,3.5,4.0,5.0,10.0])
X /= (X+3)
person = np.array([83.4,83.4,83.41,83.41,83.39,83.38,83.37,83.36,83.35,83.33,83.23])
bike = np.array([83.07,83.15,83.34,83.41,83.46,83.47,83.49,83.48,83.47,83.46,83.36])
car = np.array([83.44,83.41,83.41,83.41,83.42,83.41,83.40,83.40,83.40,83.39,83.35])
background = np.array([80.82,83.12,83.45,83.41,83.34,83.31,83.29,83.26,83.24,83.21,83.14])

plt.figure(figsize=(10,5), dpi=64, facecolor='white', edgecolor='k') 
ax2 = plt.gca()
ax2.set_facecolor('white')
ax2.plot(X, person, label='Person', linewidth=4)
ax2.plot(X, bike, label='Bike', linewidth=4)
ax2.plot(X, car, label='Car', linewidth=4)
ax2.plot(X, background, label='Background', linewidth=4)
ax2.set_xlabel('prior of a specific class', fontsize=20)
ax2.set_ylabel('AP50', fontsize=20)
ax2.grid('on')
ax2.set_xlim([0, 0.77])
ax2.scatter(1/4, 83.41, marker="^", color='c', linewidth=10, label='BayesFusion', zorder=10)
ax2.legend(loc='lower right', fontsize=20)
for tick in ax2.xaxis.get_major_ticks():
    tick.label.set_fontsize(20) 
for tick in ax2.yaxis.get_major_ticks():
    tick.label.set_fontsize(20) 
#plt.title('Varying per class prior', fontsize=20)
plt.savefig('Vary_class_prior.png', format='png', dpi=220, transparent = False, bbox_inches='tight')



X = np.array([100,1000,3000,5000,6000,7000,8000,10000,15000,20000,30000])
X = np.array([0.002682188,0.026189666,0.074658438,0.118531162,0.138943566,0.158431976,0.177057743,0.211940741,0.287449936,0.349754298,0.446541536])
#pdb.set_trace()
bayes_prior = np.array([80.88,82.93,83.22,83.28,83.28,83.28,83.28,83.27,83.24,83.21,83.18])
bayes = np.array([83.41,83.41,83.41,83.41,83.41,83.41,83.41,83.41,83.41,83.41,83.41])
plt.figure(figsize=(10,5), dpi=64, facecolor='white', edgecolor='k') 
ax2 = plt.gca()
ax2.set_facecolor('white')
ax2.plot(X, bayes_prior, label='Bayesian with prior', linewidth=4)
ax2.plot(X, bayes, label='Bayesian', linewidth=4, color='r')
ax2.set_xlabel('background prior', fontsize=20)
ax2.set_ylabel('AP50', fontsize=20)
ax2.set_xlim([0, 0.45])
ax2.grid('on')
for tick in ax2.xaxis.get_major_ticks():
    tick.label.set_fontsize(20) 
for tick in ax2.yaxis.get_major_ticks():
    tick.label.set_fontsize(20) 

ax2.legend(loc='lower right', fontsize=20)
plt.title('Bayesian with prior', fontsize=20)
plt.savefig('Bayesian_with_prior.png', format='png', dpi=220, transparent = False, bbox_inches='tight')


"""
#ax2.set_ylim([0.3, 0.65])
# ax2.legend(loc='lower center', fontsize=25)
for tick in ax2.yaxis.get_major_ticks():
    tick.label.set_fontsize(15)        
ax=ax2.twinx()
a = np.concatenate((openlikelihood_open_CF,openlikelihood_open_MN,openlikelihood_open_SV,openlikelihood_open_CS))
np.random.shuffle(a)
a = a[:len(openlikelihood_closed)]
ax.hist(openlikelihood_closed, 100, alpha=0.3, label='open-set');
ax.hist(a, 100, alpha=0.3, label='closed-set');
ax.set_ylim([0, 800])
# ax.set_ylabel('density', fontsize=25)
ax.set_yticks([])
ax.legend(loc='center', fontsize=20)
plt.title(r'(d) OpenGAN$^{fea}$: F1 vs. open likelihood threshold', fontsize=20)
for tick in ax2.xaxis.get_major_ticks():
    tick.label.set_fontsize(15) 
"""
# plt.savefig('OpenGANpix-F1.eps', format='eps', dpi=1000);

#plt.savefig('OpenGAN-F1.pdf', format='pdf', dpi=220, transparent = True, bbox_inches='tight')