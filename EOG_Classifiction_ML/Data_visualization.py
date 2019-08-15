import matplotlib.pyplot as plt
import cv2
import os
import  numpy as np
from matplotlib.ticker import NullFormatter
class Data_visualizer():

    def __init__(self):
        return

    def Visulize_Signal_On_disk(self,signal,path,signal_len,color,title):
        plt.cla()
        plt.plot([sample for sample in range(signal_len)],signal,color)
        plt.xlabel('Samples')
        plt.ylabel("Signal's value ")
        plt.title(title)
        plt.grid()
        fig = plt.gcf()
        fig.savefig(path+'/'+title)
        return
    def Visualize_Random_Data(self,Input,title='Orignal Signal'):

        fig,axs = plt.subplots(2,2)
        axs[0,0].plot([i for i in range(len(Input['up_train']))],Input['up_train'])
        axs[0,0].set_xlabel('Samples')
        axs[0,0].set_ylabel('Signal')
        axs[0,0].set_title(title+' Up train')
        axs[0,0].grid(True)

        axs[0,1].plot([i for i in range(len(Input['up_test']))], Input['up_test'])
        axs[0,1].set_xlabel('Samples')
        axs[0,1].set_ylabel('Signal')
        axs[0,1].set_title(title+' Up test')
        axs[0,1].grid(True)

        axs[1,0].plot([i for i in range(len(Input['down_train']))], Input['down_train'])
        axs[1,0].set_xlabel('Samples')
        axs[1,0].set_ylabel('Signal')
        axs[1,0].set_title(title+' Down train')
        axs[1,0].grid(True)

        axs[1,1].plot([i for i in range(len(Input['down_test']))], Input['down_test'])
        axs[1,1].set_xlabel('Samples')
        axs[1,1].set_ylabel('Signal')
        axs[1,1].set_title(title+' Down test')
        axs[1,1].grid(True)

        fig.tight_layout()

        plt.show()

        return


    def convert_single_signal_fig(self, single_signal):
        plt.cla()
        plt.plot([x for x in range(len(single_signal))], single_signal, 'r')
        plt.savefig('img')
        img = cv2.resize(cv2.imread('img.png'),(500,500))
        os.remove('img.png')
        return img

    def create_video(self,frames):

        fps = 1
        out = cv2.VideoWriter('Video.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (500, 500))
        for i in range(len(frames)):
            #out.write(frames[i][...,::-1])
            cv2.imshow('img',frames[i])
            cv2.waitKey(3000)

        cv2.destroyAllWindows()
        #out.release()

        return
