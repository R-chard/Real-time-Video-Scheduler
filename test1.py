from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager
import cv2

class SimpleClass(object):
    def __init__(self):
        
        self.var = [1,2,3]

    def set(self, i, value):
        self.var[i] = value

    def get(self):
        return self.var
        

def change_obj_value(obj):
    cap = cv2.VideoCapture("media/los_angeles.mp4")
    ret, frame = cap.read()
    obj.set(1,frame)


if __name__ == '__main__':
    BaseManager.register('SimpleClass', SimpleClass)
    manager = BaseManager()
    manager.start()
    inst = manager.SimpleClass()

    print(inst.get())          # 100

    p = Process(target=change_obj_value, args=[inst])
    p.start()
    p.join()

    print(inst)                    # <__main__.SimpleClass object at 0x10cf82350>
    print(inst.get())          # 100

    
