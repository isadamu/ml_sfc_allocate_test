import numpy as np
import sfc_model as sfc

num_i = [3,4,5]

scope_L = [[1, 1000],[1, 100],[1,100],[1,1000]]

K = 2

my_model = sfc.sfc_model(num_i = num_i, scope_L = scope_L, K = K)

def random_qos( factor = 1, N = 3, scope_L = [[1, 1000],[1, 100],[1,100],[1,1000]], K = 2 ):
    L = len(scope_L)
    qos = np.zeros(L)
    for i in range(L):
        # qos[i] = np.random.rand(1)[0] * (scope_L[i][1] * factor - scope_L[i][0]) + scope_L[i][0]
        if i < K:
            qos[i] = np.random.rand(1)[0] * (scope_L[i][1] * factor * 0.15 - scope_L[i][0]) + scope_L[i][0]
        else:
            qos[i] = np.random.rand(1)[0] * (scope_L[i][1] * N * 5 - scope_L[i][0]) + scope_L[i][0]
    return qos


##################################################

N = 3
back_muti = np.zeros(N)
category = 1
for i in range(N):
    back_muti[i] = 1
    category = category * num_i[i]
    for j in range(i+1, N):
        back_muti[i] = back_muti[i] * num_i[j]

def get_category(path):
    if path is None:
        return category
    
    cate = 0
    for i in range(N):
        cate = cate + path[i] * back_muti[i]
    
    return int(cate)

###############################################

T = 100 * 100 * 100
N = 3
num_i = [3,4,5]
samples = np.zeros((T, len(scope_L)*np.sum(num_i) + len(scope_L)*2))
# labels = np.zeros((T, len(num_i) + np.sum(num_i)))
labels = np.zeros((T, category+1))
for t in range(T):
    qos = random_qos()
    my_model.construct()
    path, _ = my_model.serveQOS(random_qos())
    sample = qos
    sample = np.hstack((sample, my_model.W))
    sample = np.hstack((sample, my_model.qoses.reshape(-1)))
    label = np.zeros(category+1)
    idx = get_category(path)
    label[idx] = 1          
    samples[t] = sample.reshape(1,-1)
    labels[t] = label.reshape(1,-1)
    
    if t % 1000 == 0:
        print(t)

np.savetxt('data/X_train_2.csv', samples, delimiter = ',')
np.savetxt('data/Y_train_2.csv', labels, delimiter = ',')