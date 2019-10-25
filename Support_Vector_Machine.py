import numpy as np
import time

class SVM:
    def __init__(self):
        print("SVM initialized")
            
    def fit(self, data):
        self.data = data
        opt_dict = {}       #{||w|| : [w,b]}
        transforms = [[1,1],
                      [1,-1],
                      [-1,1],
                      [-1,-1]]
        all_data = []
        for labels in self.data:
            for featureset in self.data[labels]:
                for feature in featureset:
                    all_data.append(feature)
                    
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        steps = [self.max_feature_value * 0.1,      #decreasing the number of steps as we go down
                 self.max_feature_value * 0.01,
                 self.max_feature_value * 0.001]

        b_range_multiple = 5
        b_multiple = 5

        latest_optimum = self.max_feature_value * 10

        for step in steps:
            w = np.array([latest_optimum,latest_optimum])            
            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
                                   self.max_feature_value*b_range_multiple,
                                   step*b_multiple):
                    for transformation in transforms:
                        w_t = w * transformation
                        found_option = True
                        for i in self.data:
                            for xi in self.data[i]:         #constraint: yi * (xi . w + b) >= 1
                                yi = i
                                if not yi * (np.dot(w_t,xi) + b) >=1:
                                    found_option = False
                                #print(xi,',',yi * (np.dot(w_t,xi) + b))
                                
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b]
                if w[0] < 0:
                    optimized = True
                    #print('Optimized a step')
                else:
                    w = w - step
                    
            
            norms = sorted([n for n in opt_dict])
            #print(norms)
            opt_choice = opt_dict[norms[0]]
            #print(opt_choice)
            self.w = opt_choice[0]            
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step * 2
                
                
    def predict(self, feature):
        prediction = np.sign(np.dot(np.array(feature),self.w) + self.b)        
        return prediction      




#-1, 1 are two classes
data = {
        -1: np.array([
            [1,7],
            [2,8],
            [3,8]]),
        1: np.array([
            [5,1],
            [6,-1],
            [7,3]])
        }

pred = [
        [0,10],
        [1,3],
        [3,4],
        [3,5],
        [5,5],
        [5,6],
        [6,-5],
        [5,8]]




t = time.time()
clfr = SVM()
clfr.fit(data)
for p in pred:
    print("Feature: ",p,'-> Predicted Label:',clfr.predict(p))
    
print("Time taken:",time.time()-t)







input()







