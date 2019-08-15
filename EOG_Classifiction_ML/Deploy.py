import  Data_visualization as dv
import Preprocessing_Signal as Pre_process
import Data_Loader
import  numpy as np
import os
import cv2
import time
import DSP_SVM as svm_c
import random
import DSP_KNN
import tkinter as tk
from tkinter import messagebox
from PIL import  Image, ImageDraw, ImageFont
from tkinter import ttk
from PIL import Image,ImageTk
from tkinter import filedialog
Wavelet_Level_Control = 6  # 6 for KNN and 1 for SVM
BandPass_Level_Control = 1 # 1 for KNN and 6 for SVM
Neighborhood_KNN_Control = 1
Svm_Kernel_Control = 'linear'
Svm_C_Control = 100
Svm_Gamma_Contro = 0.0001
class GUI_Data_Interaction():

    def __init__(self):

        self.Pre_process_obj = Pre_process.Preprocessing()
        self.obj_data_loader = Data_Loader.Data_Reader()
        self.visualize_obj=dv.Data_visualizer()

        self.frames = []
        self.wavelet_names=['db1','db2','db3','db4']
        self.svm = svm_c.Hero_SVM()
        self.KNN = DSP_KNN.KNN()
        self.flag_load_data = False
        self.flag_preprocessing = False
        self.wavelet_name='db1'

        self.svm_classify = False
        self.knn_classsify = False
        return

    def Load_Data(self):
        self.flag_load_data = False
        try:
            (self.train_data_signals, self.train_data_lbls), (self.test_data_signals, self.test_data_lbls) = self.obj_data_loader.Load_Data()
            self.train_data_features = []
            self.test_data_features = []
            self.flag_load_data = True
        except:
            print("Error in Load_Data Function in GUI_Data_Interaction Class")
        return

    def preprocessing(self):
        self.flag_preprocessing = False
        print('self.wavelet_name',self.wavelet_name)
        try:
            for i in range(len(self.train_data_signals)):
                self.train_data_signals[i] = self.Pre_process_obj.Mean_Removal(self.train_data_signals[i],self.train_data_signals[i].shape[0])
                self.train_data_signals[i] = self.Pre_process_obj.BandPass(self.train_data_signals[i], 0.5, 20, 176, BandPass_Level_Control)
                self.train_data_signals[i] = self.Pre_process_obj.Normalize(self.train_data_signals[i], self.train_data_signals[i].shape[0])

                self.train_data_features.append(self.Pre_process_obj.feature_Extraction(self.train_data_signals[i],Wavelet_Level_Control,self.wavelet_name))
                self.train_data_features[i] = self.Pre_process_obj.Mean_Removal(np.array(self.train_data_features[i]),np.array(self.train_data_features[i].shape[0]))
                self.train_data_features[i]=self.Pre_process_obj.Normalize(np.array(self.train_data_features[i]),np.array(self.train_data_features[i]).shape[0])

            for i in range(len(self.test_data_signals)):
                self.test_data_signals[i] = self.Pre_process_obj.Mean_Removal(self.test_data_signals[i],self.test_data_signals[i].shape[0])
                self.test_data_signals[i] = self.Pre_process_obj.BandPass(self.test_data_signals[i], 0.5, 20, 176, BandPass_Level_Control)
                self.test_data_signals[i] = self.Pre_process_obj.Normalize(self.test_data_signals[i],self.test_data_signals[i].shape[0])

                self.test_data_features.append(self.Pre_process_obj.feature_Extraction(self.test_data_signals[i], Wavelet_Level_Control,self.wavelet_name))
                self.test_data_features[i] = self.Pre_process_obj.Mean_Removal(np.array(self.test_data_features[i]),np.array(self.test_data_features[i].shape[0]))
                self.test_data_features[i]=self.Pre_process_obj.Normalize(np.array(self.test_data_features[i]),np.array(self.test_data_features[i]).shape[0])

            self.train_data_features = np.array(self.train_data_features)
            self.test_data_features = np.array(self.test_data_features)

            print("The shape of train_data_features : ", self.train_data_features.shape)
            print("The shape of train_data_lbls : ", self.train_data_lbls.shape)
            print("The shape of test_data_features : ", self.test_data_features.shape)
            print("The shape of test_data_lbls : ", self.test_data_lbls.shape)



            self.flag_preprocessing = True
        except:
            print("Error in preprocessing Function in GUI_Data_Interaction Class")
        return

    def SVM_classification(self):

        try:
            index_list = [i for i in range(self.train_data_features.shape[0])]
            random.shuffle(index_list)
            self.train_data_features=self.train_data_features[index_list]
            self.train_data_lbls=self.train_data_lbls[index_list]
            self.train_data_signals = self.train_data_signals[index_list]

            print("*****************************************************************")
            print("-->> Features : ")
            self.Model_svm = self.svm.Create_Model(Svm_Kernel_Control,Svm_C_Control,Svm_Gamma_Contro)
            self.Model_svm = self.svm.Train_Model(self.train_data_features, np.reshape(self.train_data_lbls,self.train_data_lbls.shape[0])
                                              ,self.Model_svm)
            Acc = self.svm.Calculate_Accuracy(self.train_data_features, np.reshape(self.train_data_lbls,self.train_data_lbls.shape[0]),
                                               self.Model_svm)
            print("The Accuracy of train using SVM is : ", Acc)
            Acc_test = self.svm.Calculate_Accuracy(self.test_data_features, np.reshape(self.test_data_lbls,self.test_data_lbls.shape[0]),
                                                    self.Model_svm)
            print("The Accuracy of test using SVM  is : ", Acc_test)
            messagebox.showinfo('Info','The Test Accuracy of SVM is : '+str(Acc_test))

            print("*****************************************************************")

            print("*****************************************************************")
            print("-->> Original Signal : ")
            self.Model_svm_org = self.svm.Create_Model(Svm_Kernel_Control, Svm_C_Control, Svm_Gamma_Contro)
            self.Model_svm_org = self.svm.Train_Model(self.train_data_signals,
                                              np.reshape(self.train_data_lbls, self.train_data_lbls.shape[0])
                                              , self.Model_svm_org)
            Acc = self.svm.Calculate_Accuracy(self.train_data_signals,
                                              np.reshape(self.train_data_lbls, self.train_data_lbls.shape[0]),
                                              self.Model_svm_org)
            print("The Accuracy of train using SVM is : ", Acc)
            Acc_test = self.svm.Calculate_Accuracy(self.test_data_signals,
                                                   np.reshape(self.test_data_lbls, self.test_data_lbls.shape[0]),
                                                   self.Model_svm_org)
            print("The Accuracy of test using SVM  is : ", Acc_test)
            print("*****************************************************************")
            self.svm_classify = True
        except:
            messagebox.showinfo('Info','Error in the SVM_classification function in GUI_Data_Interaction class')
        return
    def KNN_classification(self):
        try:
            index_list = [i for i in range(self.train_data_features.shape[0])]
            random.shuffle(index_list)
            self.train_data_features=self.train_data_features[index_list]
            self.train_data_lbls=self.train_data_lbls[index_list]
            self.train_data_signals = self.train_data_signals[index_list]
            print("*****************************************************************")
            print("-->> Features : ")
            self.Knn_Model = self.KNN.fit(self.train_data_features, np.reshape(self.train_data_lbls,self.train_data_lbls.shape[0]),Neighborhood_KNN_Control)
            Acc_knn = self.KNN.calc_Acc(self.train_data_features, np.reshape(self.train_data_lbls,self.train_data_lbls.shape[0]),self.Knn_Model)
            print("The Accuracy of train using KNN is : ", Acc_knn)
            Acc_knn = self.KNN.calc_Acc(self.test_data_features, np.reshape(self.test_data_lbls, self.test_data_lbls.shape[0]), self.Knn_Model)
            print("The Accuracy of test using KNN is : ", Acc_knn)
            messagebox.showinfo('Info','The Test Accuracy of KNN is : '+str(Acc_knn))
            print("*****************************************************************")


            print("*****************************************************************")
            print("-->> Original Signal : ")
            self.Knn_Model_org = self.KNN.fit(self.train_data_signals, np.reshape(self.train_data_lbls,self.train_data_lbls.shape[0]),Neighborhood_KNN_Control)
            Acc_knn = self.KNN.calc_Acc(self.train_data_signals, np.reshape(self.train_data_lbls,self.train_data_lbls.shape[0]),self.Knn_Model_org)
            print("The Accuracy of train using KNN is : ", Acc_knn)
            Acc_knn = self.KNN.calc_Acc(self.test_data_signals, np.reshape(self.test_data_lbls, self.test_data_lbls.shape[0]), self.Knn_Model_org)
            print("The Accuracy of test using KNN is : ", Acc_knn)
            print("*****************************************************************")
            self.knn_classsify = True

        except:
            messagebox.showinfo('Info','Error in the KNN_classification function in GUI_Data_Interaction class')

        return
    def Visualize_random_data(self):
        org_data={'up_train':self.train_data_signals[6],'up_test':self.test_data_signals[0],'down_train':self.train_data_signals[25],'down_test':self.test_data_signals[7]}
        features_data={'up_train':self.train_data_features[6],'up_test':self.test_data_features[0],'down_train':self.train_data_features[25],'down_test':self.test_data_features[7]}
        self.visualize_obj.Visualize_Random_Data(org_data)
        self.visualize_obj.Visualize_Random_Data(features_data,'Features of Signal')

        return
    def Life_Cycle_Process(self,Signal_path):

        dimage = Image.new('RGB', (500, 500), (0, 0, 0))
        draw = ImageDraw.Draw(dimage)
        Font = ImageFont.truetype("arial.ttf", 20)
        draw.text((150,250), 'Original Signal', fill=(255, 255, 255), font=Font)
        self.frames.append(np.array(dimage))

        single_signal = np.array(self.obj_data_loader.Read_Single_Signal(Signal_path))


        img1 = self.visualize_obj.convert_single_signal_fig(single_signal)

        self.frames.append(img1)



        single_signal_mean_removed = self.Pre_process_obj.Mean_Removal(single_signal,
                                                                      single_signal.shape[0])
        img2 = self.visualize_obj.convert_single_signal_fig(single_signal_mean_removed)

        dimage = Image.new('RGB', (500, 500), (0, 0, 0))
        draw = ImageDraw.Draw(dimage)
        Font = ImageFont.truetype("arial.ttf", 20)
        draw.text((150, 250), 'Signal After Mean Removing', fill=(250, 250, 250), font=Font)
        self.frames.append(np.array(dimage))

        self.frames.append(img2)


        single_signal_mean_removed_bandpassed = self.Pre_process_obj.BandPass(single_signal_mean_removed, 0.5, 20, 176,
                                                                   BandPass_Level_Control)
        img3 = self.visualize_obj.convert_single_signal_fig(single_signal_mean_removed_bandpassed)
        dimage = Image.new('RGB', (500, 500), (0, 0, 0))
        draw = ImageDraw.Draw(dimage)
        Font = ImageFont.truetype("arial.ttf", 20)
        draw.text((150, 250), 'Filtered By Bandpass', fill=(225,225, 255), font=Font)
        self.frames.append(np.array(dimage))
        self.frames.append(img3)


        single_signal_mean_removed_bandpassed_normalized = self.Pre_process_obj.Normalize(single_signal_mean_removed_bandpassed,
                                                                    single_signal_mean_removed_bandpassed.shape[0])
        img4 = self.visualize_obj.convert_single_signal_fig(single_signal_mean_removed_bandpassed_normalized)
        dimage = Image.new('RGB', (500, 500), (0, 0, 0))
        draw = ImageDraw.Draw(dimage)
        Font = ImageFont.truetype("arial.ttf", 20)
        draw.text((150, 250), 'After Normalization', fill=(255, 255, 255), font=Font)
        self.frames.append(np.array(dimage))
        self.frames.append(img4)

        single_signal_mean_removed_bandpassed_normalized_feature_Extracted= self.Pre_process_obj.feature_Extraction(single_signal_mean_removed_bandpassed_normalized
                                                                                                                    , Wavelet_Level_Control, self.wavelet_name)
        img5 = self.visualize_obj.convert_single_signal_fig(single_signal_mean_removed_bandpassed_normalized_feature_Extracted)
        dimage = Image.new('RGB', (500, 500), (0, 0, 0))
        draw = ImageDraw.Draw(dimage)
        Font = ImageFont.truetype("arial.ttf", 20)
        draw.text((150,250), 'After Feature Extraction', fill=(255, 255, 255), font=Font)
        self.frames.append(np.array(dimage))
        self.frames.append(img5)
        self.visualize_obj.create_video(self.frames)
        return


class GUI():

    def __init__(self,Master,Gui_data_interaction_obj):
        self.root=Master
        self.Gui_data_interaction = Gui_data_interaction_obj
        return

    def Button_Load_Data(self):
        button=tk.Button(master=self.root,text='Load Data',command=self.cmd_Button_Load_Data,width=32)
        button.place(x=118,y=20)
        return
    def cmd_Button_Load_Data(self):
        self.Gui_data_interaction.Load_Data()
        if self.Gui_data_interaction.flag_load_data:
            messagebox.showinfo('Info','Data loaded correctly')
        else:
            messagebox.showinfo('Info', 'Error in Load_Data Function in GUI_Data_Interaction Class')
        return

    def Button_Preprocesing(self):
        button=tk.Button(master=self.root,text='preprocessing',command=self.cmd_Button_Preprocesing,width=32)
        button.place(x=118,y=120)
        return

    def cmd_Button_Preprocesing(self):
        self.Gui_data_interaction.wavelet_name=self.Gui_data_interaction.wavelet_names[self.combobox.current()]
        self.Gui_data_interaction.preprocessing()
        if self.Gui_data_interaction.flag_preprocessing:
            messagebox.showinfo('Info','Data preprocessing done correctly')
        else:
            messagebox.showinfo('Info', 'Error in Load_DataFunction in preprocessing Class')
        return

    def Button_Svm(self):
        button=tk.Button(master=self.root,text='SVM classifier',command=self.cmd_Button_Svm,width=32)
        button.place(x=118,y=170)
        return
    def cmd_Button_Svm(self):
        self.Gui_data_interaction.SVM_classification()
        return

    def Button_Knn(self):
        button=tk.Button(master=self.root,text='KNN classifier',command=self.cmd_Button_Knn,width=32)
        button.place(x=118,y=220)
        return
    def cmd_Button_Knn(self):
        self.Gui_data_interaction.KNN_classification()
        return
    def Button_plot_random_data(self):
        button=tk.Button(master=self.root,text='Visualize Random Data',width=32,command=self.cmd_Button_plot_random_data)
        button.place(x=118,y=270)
        return

    def cmd_Button_plot_random_data(self):
        self.Gui_data_interaction.Visualize_random_data()
        return

    def Combobox_wavelet_name(self):
        self.combobox=ttk.Combobox(master=self.root,values=self.Gui_data_interaction.wavelet_names,width=32)
        self.combobox.current(0)
        self.combobox.place(x=118,y=70)
        return
    def Background(self):
        background = Image.open('ESAGaiaMilkyWay.jpg')
        background = background.resize((500, 400),Image.ANTIALIAS)
        background = ImageTk.PhotoImage(background)
        bg_lbl = tk.Label(master=self.root,image=background)
        bg_lbl.place(x=0, y=0, relwidth=1, relheight=1)
        bg_lbl.image = background
        return
    def Button_Test_and_Visulize_Life_Cylce(self):
        button = tk.Button(master=self.root, text='Signal Life Cycle', width=32, command=self.cmd_Button_Test_and_Visulize_Life_Cylce)
        button.place(x=118, y=320)
        return

    def cmd_Button_Test_and_Visulize_Life_Cylce(self):
        if self.Gui_data_interaction.knn_classsify ==False and self.Gui_data_interaction.svm_classify == False:
            messagebox.showinfo('Info','First Classify by svm and knn')
        else:
            messagebox.showinfo('Info','You will see a story video about the signal life cycle')
            self.Gui_data_interaction.frames = []
            Signal_path = filedialog.askopenfilename(initialdir ='Data',filetypes=(("File", ".ds"), ("File", ".txt"), ("All Files", "*.ds")))
            self.Gui_data_interaction.Life_Cycle_Process(Signal_path)
        return
if __name__=='__main__':

    start=time.time()

    master_roo=tk.Tk()
    master_roo.resizable(0,0)
    Gui_data_interaction_obj = GUI_Data_Interaction()
    Gui_obj = GUI(master_roo,Gui_data_interaction_obj)

    master_roo.title("EOG UP and Down")
    master_roo.minsize(500,400)
    Gui_obj.Background()
    Gui_obj.Button_Load_Data()
    Gui_obj.Combobox_wavelet_name()
    Gui_obj.Button_Preprocesing()
    Gui_obj.Button_Svm()
    Gui_obj.Button_Knn()
    Gui_obj.Button_plot_random_data()
    Gui_obj.Button_Test_and_Visulize_Life_Cylce()
    master_roo.mainloop()


    end =time.time()
    print("The total time is : ",end-start)