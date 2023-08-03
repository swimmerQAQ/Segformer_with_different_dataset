import json
import numpy as np
class LMS:
    def __init__(self,map_goal_name,map_goal_path):
        #核对表,即为最终目标 标签分类
        tfr = open(map_goal_path, "r")
        temp = json.load(tfr)
        self.map_goal = temp[map_goal_name]

        # label映射终点 dict['type name' : number]
        self.remapped_label = {}

        # label映射起点
        self.remapping_label = {}

        # 映射规则
        self.map_strategy = {}

        # 
        self.map_array = {}
        pass

    def read_mapping(self,map_name,map_path):
        tfr = open(map_path, "r")
        temp = json.load(tfr)
        self.remapping_label[map_name] = temp[map_name]
    def read_strategy(self,map_strategy_name,map_strategy_path):
        tfr = open(map_strategy_path, "r")
        temp = json.load(tfr)
        self.map_strategy[map_strategy_name] = temp[map_strategy_name]
        pass
    def show_ms(self):
        for key in self.map_strategy :
            print(key,"\n",self.map_strategy[key],"\n")
        pass
    def show_remapping(self):
        for key in self.remapping_label :
            print(key,"\n",self.remapping_label[key],"\n")
        return self.remapping_label
    def show_map_goal(self):
        print("\n","map_goal\n",self.map_goal,"\n")
        pass
    def show_mapped(self):
        for strategy in self.remapped_label :
            print("\ndata type:",strategy)
            for key in self.remapped_label[strategy] :
                print(key,"=",self.remapped_label[strategy][key])
        pass
    def map_action(self):
        for strategy in self.map_strategy:                  #执行所有的映射策略
            temp_strategy = self.map_strategy[strategy]     #为了使用字典
            data_type = temp_strategy['origin']             #规定 origin字
            self.remapped_label[data_type] = {}
            store = self.remapped_label[data_type]          #提前申请使用中间量store
            ####伴随生成 映射数组 减一是因为 origin字
            map_array = np.empty(len(temp_strategy)-1,dtype = int, order = 'C')
            num = 0
            for one_type in temp_strategy:                  #策略数量 和 源数据相当
                ori = one_type                              #当前data type（源数据） 的类别
                des = temp_strategy[one_type]               #映射目标中的值
                if not des in self.map_goal:                #排除origin：在目标中没有该 字
                    if ori != 'origin':
                        store[ori] = 255                    #存在非目标标签中的类，划归为255
                        # print("the type =" , one_type ,num," set 255")
                        map_array[num] = 255
                        num += 1
                    continue
                number = self.map_goal[des]                 #获得目标标签号码
                store[ori] = number                         #存储 源数据 映射到目标标签 的数字
                #####给定数组
                map_array[num] = number
                num += 1
        self.map_array[temp_strategy["origin"]] = np.asarray(map_array, dtype=np.uint8)
    '''
    return the mapped label dtype = numpy.array
    '''
    def get_array(self) :
        self.map_action()
        # self.show_mapped()
        # for map_array_type in self.map_array:
        #     print(map_array_type , self.map_array[map_array_type])
        return self.map_array
