import numpy as np

class Data_Reader():
    def __init__(self):

        self.Data_sub_folders = ['train_up.txt','train_down.txt','test_up.txt','test_down.txt']
        self.Main_path='Data\{}'
        self.train_up_path = self.Main_path.format(self.Data_sub_folders[0])
        self.train_down_path = self.Main_path.format(self.Data_sub_folders[1])
        self.test_up_path = self.Main_path.format(self.Data_sub_folders[2])
        self.test_down_path = self.Main_path.format(self.Data_sub_folders[3])
        return

    def Read_Single_Signal(self,signal_path):

        signal_file = open(signal_path)
        singal_data_str = signal_file.readlines()
        singal_data_str = singal_data_str[0].strip().split()
        singal_data = [float(val) for val in singal_data_str]

        return singal_data

    def Read_Data(self , Data_Type = 'train'):

        path_up = ''
        path_down = ''

        if Data_Type == 'train':
            path_up = self.train_up_path
            path_down = self.train_down_path

        elif Data_Type == 'test':
            path_up = self.test_up_path
            path_down = self.test_down_path

        else:
            print("**********************************************************************************************")
            print("Error Input Value for Data_Type Parameter of the Function <<Read_Data>> in the folder <<Data.py>> ")
            print("The Input value of Data_Type is : {}".format(Data_Type))
            print("Please Note that the values that Data_type can take are train or test ")
            print("**********************************************************************************************")
            return

        file_up_data = open(path_up)
        file_down_data = open(path_down)

        up_data_as_strings = file_up_data.readlines()
        dowm_data_as_strings = file_down_data.readlines()

        for i  in range(len(up_data_as_strings)):
            up_data_as_strings[i] = up_data_as_strings[i].strip().split()

        for i in range(len(dowm_data_as_strings)):
            dowm_data_as_strings[i] = dowm_data_as_strings[i].strip().split()

        up_data_as_numeric=[]
        dowm_data_as_numeric=[]

        for row in up_data_as_strings:
            up_data_as_numeric_row=[]
            for value_string in row:
                up_data_as_numeric_row.append(float(value_string))

            up_data_as_numeric.append(up_data_as_numeric_row)

        for row in dowm_data_as_strings:
            dowm_data_as_numeric_row=[]
            for value_string in row:
                dowm_data_as_numeric_row.append(float(value_string))

            dowm_data_as_numeric.append(dowm_data_as_numeric_row)



        return  up_data_as_numeric , dowm_data_as_numeric

    def Load_Data(self):

        up_data_train , down_data_train = self.Read_Data('train')
        up_data_test , down_data_test = self.Read_Data('test')

        # the label 1 is for up
        # tha label 0 is fro down

        train_data_signals = []
        test_data_signals = []

        train_data_lbls = []
        test_data_lbls = []

        for signal in up_data_train:
            train_data_signals.append(signal)
            train_data_lbls.append(1)

        for signal in down_data_train:
            train_data_signals.append(signal)
            train_data_lbls.append(-1)

        for signal in up_data_test:
            test_data_signals.append(signal)
            test_data_lbls.append(1)

        for signal in down_data_test:
            test_data_signals.append(signal)
            test_data_lbls.append(-1)

        return ( np.array(train_data_signals) , np.array(train_data_lbls).reshape(len(train_data_lbls),1) ) , ( np.array(test_data_signals) , np.array(test_data_lbls).reshape(len(test_data_lbls),1) )




