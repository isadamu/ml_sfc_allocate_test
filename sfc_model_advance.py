# 模型生成函数，负责依据参数生成sfc模型。

class sfc_model:

    import numpy as np
    
    # 网格大小
    max_shape = [5,5]

    # Qos 的属性个数
    L = 4

    # 每一个属性的取值范围
    scope = [1, 100]
    
    
    vnf_count = None # 每一类VNF的个数
    vnf_qos = None # Qos矩阵
    mask = None # 所需的Qos属性
    req = None # Qos请求
    w_qos = None # Qos权重
    k_qos = None # 前后向Qos

    # 构造函数
    def __init__(self, max_shape = [5, 5], L = 4, scope = [1, 100]):
        self.max_shape = max_shape
        self.L = L
        self.scope = scope
    
    
    ######################################################
    '''
    第一大步，随机出所有需要的参数：

    第一步，确定每一类VNF的个数。
    第二步，随机出一个Qos请求，随机出每一个Qos属性的权重。
    第三步，确定出需要使用的Qos属性，不需要的属性对应的取值置为0。
    第四步，确定Qos属性的前向以及后向性。
    '''
    def __random_all(self):
        vnf_count = self.np.random.randint(self.max_shape[1] + 1, size = self.max_shape[0])
        while self.np.sum(vnf_count) == 0:
            vnf_count = self.np.random.randint(self.max_shape[1] + 1, size = self.max_shape[0])
        
        vnf_qos = self.np.random.rand(self.max_shape[0], self.max_shape[1], self.L) * (self.scope[1] - self.scope[0]) + self.scope[0]
        
        for i in range(len(vnf_count)):
            for j in range(self.max_shape[1], vnf_count[i], -1):
                vnf_qos[i,j-1,:] = 0
        
        mask = self.np.random.rand(self.L) < 0.3
        while self.np.sum(mask) == self.L:
            mask = self.np.random.rand(self.L) < 0.3
        
        vnf_qos[:,:,mask] = 0
        
        req = self.np.random.rand(self.L) * (self.scope[1] * self.max_shape[0] * 3 - self.scope[0]) + self.scope[0]
        req[mask] = 0
        
        w_qos = self.np.random.rand(self.L)
        w_qos[mask] = 0
        w_qos = w_qos / self.np.sum(w_qos)
        
        k_qos = self.np.ones(self.L, dtype = self.np.int)
        k_qos[self.np.random.rand(self.L) < 0.5] = -1
        
        for i in range(len(req)):
            if k_qos[i] == 1:
                req[i] = req[i] / (self.max_shape[0] * 5)
        
        self.vnf_count = vnf_count
        self.vnf_qos = vnf_qos
        self.mask = mask
        self.req = req
        self.w_qos = w_qos
        self.k_qos = k_qos
        
        
    ######################################################    
    '''
    第二大步，暴力求解最优解：

    '''
    # 每一个 VNF 的 utility
    U = None

    # 最优路径的 utility
    best_utility = None

    # 最优路径
    best_path = None

    # 供 __computerU 使用，计算一个 VNF 的 utility
    def __computerPerU(self, qos):
        utility = 0.0
        for i in range(len(qos)):
            score = 0.0
            if self.k_qos[i] == 1:
                score = self.w_qos[i] * (qos[i] - self.scope[0]) / (self.scope[1] - self.scope[0])
            else:
                score = self.w_qos[i] * (self.scope[1] - qos[i]) / (self.scope[1] - self.scope[0])
            utility = utility + score
        return utility
    
    # 计算每一个 VNF 的 utility , 存入表 U
    def __computerU(self):
        self.U = self.np.zeros((self.max_shape[0], self.max_shape[1]))
        for i in range(self.max_shape[0]):
            for j in range(self.max_shape[1]):
                self.U[i,j] = self.__computerPerU(self.vnf_qos[i,j,:])
            

    # dfs到所有的路径，计算路径是否比当前最优值更优
    def __dfsPath(self, path, idx):
        if idx == self.max_shape[0]: # 路径结束
            
            this_qos = self.np.zeros(self.L)
            utility = 0.0
            for i in range(len(path)):
                if path[i] == self.max_shape[1]:
                    continue
                this_qos = this_qos + self.vnf_qos[i,path[i],:]
                utility = utility + self.U[i,path[i]]
            
            flag = True
            for i in range(self.L):
                if self.k_qos[i] == 1:
                    if this_qos[i] < self.req[i]:
                        flag = False
                        break
                else:
                    if this_qos[i] > self.req[i]:
                        flag = False
                        break
            
            if flag is True and ((self.best_utility is None) or (utility > self.best_utility)):
                self.best_utility = utility
                self.best_path = path
                
            return
        
        if self.vnf_count[idx] == 0:
            if path is None:
                path = [self.max_shape[1],]
            else:
                path.append(self.max_shape[1])
            self.__dfsPath(path, idx + 1)
        
        else:
            for i in range(self.vnf_count[idx]):
                path_copy = None
                if path is None:
                    path_copy = [i,]
                else:
                    path_copy = path.copy()
                    path_copy.append(i)
                self.__dfsPath(path_copy, idx + 1)
    
    
    def next_sample(self):
        self.__random_all()
        self.__computerU()
        self.best_path = None
        self.best_utility = None
        self.__dfsPath(None, 0)
        Y = self.np.zeros((self.max_shape[0], self.max_shape[1] + 1))
        if self.best_path is None:
            Y[:,self.max_shape[1]] = 1
        else:
            for i in range(self.max_shape[0]):
                Y[i,self.best_path[i]] = 1
        w_matrix = self.np.ones((self.max_shape[0], self.max_shape[1], self.L), dtype=self.np.int) * self.w_qos.reshape(1,1,-1)
        w_matrix = w_matrix * self.k_qos.reshape(1,1,-1)
        req_matrix = self.np.ones((self.max_shape[0], self.max_shape[1], self.L), dtype=self.np.int) * self.req.reshape(1,1,-1)
        X = self.np.dstack((self.vnf_qos,w_matrix,req_matrix))
        
        return X, Y
