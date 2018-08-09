'''
模型生成函数，负责依据参数生成sfc模型。
'''

class sfc_model:

    import numpy as np
    # 每一类 VNF 的个数
    num_i = [2, 3]

    # VNF 的种类数
    N = 2

    # 每一个 VNF 的属性个数
    L = 2

    # 每一个属性的取值范围
    scope_L = [[1, 1000],[1, 100]]

    # 正向属性的个数 
    K = 1

    # 属性的权重
    W = None

    # 所有 VNF 的 qos
    qoses = None

    # 构造函数
    def __init__(self, num_i = [2, 3], scope_L = [[1, 1000],[1, 100]], K = 1):
      self.num_i = num_i
      self.L = len(scope_L)
      self.scope_L = scope_L
      self.K = K
      self.N = len(num_i)
    
    # 随机出各个 VNF 的 qos 以及每个属性的权重
    def construct(self):
        qoses = []
        total = self.np.sum(self.num_i);
        for i in range(total):
            vnf = []
            for j in range(self.L):
                val = self.np.random.rand(1)[0] * (self.scope_L[j][1] - self.scope_L[j][0]) + self.scope_L[j][0]
                vnf.append(val)
            
            qoses.append(vnf)
        
        qoses = self.np.array(qoses)
        self.qoses = qoses

        W = self.np.random.rand(self.L)
        W = W / self.np.sum(W)
        self.W = W

        return qoses


    def to_file(self, path = "model.txt"):
        self.np.savetxt(path, self.qoses)

    ######################################################
    '''
    暴力求解法：
        计算出所有路径，选择出最好的
    '''
    # 每一个 VNF 的 utility
    U = None

    # 路径按照 utility 排序后的 qos
    best_utility = None

    # 按照 utility 排序后的 路径
    best_path = None

    # 供 __computerU 使用，计算一个 VNF 的 utility
    def __computerPerU(self, qos):
        utility = 0.0
        for i in range(len(qos)):
            score = 0.0
            if i < self.K:
                score = self.W[i] * (qos[i] - self.scope_L[i][0]) / (self.scope_L[i][1] - self.scope_L[i][0])
            else:
                score = self.W[i] * (self.scope_L[i][1] - qos[i]) / (self.scope_L[i][1] - self.scope_L[i][0])
            utility = utility + score
        return utility
    
    # 计算每一个 VNF 的 utility , 存入表 U
    def __computerU(self):
        self.U = self.np.zeros(self.qoses.shape[0])
        for i in range(len(self.U)):
            self.U[i] = self.__computerPerU(self.qoses[i])

    # dfs到所有的路径，计算路径是否比当前最优值更优
    def __dfsPath(self, path, idx, req_qos):
        if idx == self.N:

            path_map = []
            for i in range(len(path)):
                path_map.append(self.__mapVNFIdx(i, path[i]))

            this_qos = self.np.zeros(self.L)
            utility = 0.0
            for i in path_map:
                this_qos = this_qos + self.qoses[i]
                utility = utility + self.U[i]
            
            flag = True
            for i in range(self.L):
                if i < self.K:
                    if this_qos[i] < req_qos[i]:
                        flag = False
                        break
                else:
                    if this_qos[i] > req_qos[i]:
                        flag = False
                        break
            
            if flag is True and ((self.best_utility is None) or (utility > self.best_utility)):
                self.best_utility = utility
                self.best_path = path
               
            return
        
        for i in range(self.num_i[idx]):
            path_copy = None
            if path is None:
                path_copy = [i,]
            else:
                path_copy = path.copy()
                path_copy.append(i)
            self.__dfsPath(path_copy, idx + 1, req_qos)
    
    def __mapVNFIdx(self, i, j):
        idx = 0
        for ii in range(i):
            idx = idx + self.num_i[ii]
        return idx + j

    # 主要的函数
    def serveQOS(self, req):
        self.best_path = None
        self.best_utility = None
        self.__computerU()
        self.__dfsPath(None, 0, req)

        return self.best_path, self.best_utility

    #########################################################
        





