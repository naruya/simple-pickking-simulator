import abc
import cv2
import random
import numpy as np
import datetime
from collections import OrderedDict
from copy import deepcopy

from utils import layerize, overlay

# global座標に直したい

class OB(object, metaclass=abc.ABCMeta):
    def __init__(self, name, world=None, size=None, pos=None):
        self.name = name
        self.world = world
        self.size = size
        self.pos = pos
        self.h, self.w = size
        self.x, self.y = pos
    
    @abc.abstractmethod
    def render(self):
        pass
        
    @abc.abstractmethod
    def step(self):
        pass

class Shelf(OB):
    def __init__(self, name, world):
        super(Shelf, self).__init__(name, world, (600,1000), (350, 600))
        self.texture = cv2.imread("src/shelf.png", -1)
        
        self.num_cans = 6 # random.randint(6,9)
        self.pos1 = [self.x - self.h/2, self.y - self.w/2] # 150, 
        self.cans = self.can_generator()
    
    def get_cans(self):
        cans = deepcopy(self.cans)
        for can in cans:
            can.x += int(self.pos1[0])
            can.y += int(self.pos1[1])
        return cans
        
    def render(self):
        layer = self.texture.copy()
        for sub_ob in self.cans:
            sub_layer = layerize((self.h, self.w), (sub_ob.x, sub_ob.y), sub_ob.texture.copy())
            layer = overlay(layer, sub_layer)
        return layer
        
    def step(self):
        self.render()
    
    def can_generator(self):
        while True:
            cans = [self.Can(self) for _ in range(self.num_cans)]
            layer = self.texture.copy()
            for sub_ob in cans:
                texture = cv2.resize(sub_ob.texture.copy(), (100, 100)) # for ease
                sub_layer = layerize((self.h, self.w), (sub_ob.x, sub_ob.y), texture)
                layer = overlay(layer, sub_layer)
            layer = cv2.cvtColor(layer, cv2.COLOR_BGRA2GRAY)
            ret, img_binary= cv2.threshold(layer, 128, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if self.num_cans == len(contours) - 1: # exclude outer frame
                break
        
        target_id = np.array([can.x for can in cans]).argmin()
        if not target_id == 0:
            target = cans[target_id]
            cans[target_id] = cans[0]
            cans[0] = target
        
        cans[0].texture[:,:,:3] = [255, 63, 63]
        for i in range(1, self.num_cans):
            cans[i].texture[:,:,:3] = [63, 63, 255]
        return cans
    
    class Can():
        def __init__(self, shelf):
            self.size = 50 # random.randint(40,45)
            self.x = random.randint(self.size, shelf.h//2 - self.size)
            self.y = random.randint(self.size*2, shelf.w - self.size*2)
            texture = np.ones((self.size, self.size, 4)).astype(np.uint8) * 255
            texture[:,:,:3] = 0
            self.texture = texture 


class Robot(OB):
    def __init__(self, name, world):
        super(Robot, self).__init__(name, world, (160,160), (500, 600))
        self.texture = cv2.imread("src/robot.png", -1)
        self.state = 0
        self.action_holder = []
        
    def render(self):
        return self.texture
    
    def _report(self):
        stamp = str(datetime.datetime.now()).split()[1]
        print("[I {} {}]".format(stamp, self.name), \
              "state:", self.state)

    def _flow(self):
        if self.state == 0:
            self.state = 1
            return "skip"
        elif self.state == 1:
            target = self._check("1")
            if target == None:
                self.state = 2
                return None
            else:
                return target
        pass
    
    def _check(self, key):
        if key == "1": # object(s) before target
            cans = self.world.shelf.get_cans()
            next_target_x = 0
            next_target_id = 0
            flag = False
            for i, can in enumerate(cans):
                if i == 0:
                    continue
                if can.y-100 < cans[0].y and cans[0].y < can.y+100:
                    if next_target_x < can.x:
                        next_target_x = can.x
                        next_target_id = i
                        next_target = can
                        flag = True
            if not flag:
                return None
            else:
                return (next_target_id, next_target)

    def _run(self, info):
        if info == "skip":
            return 0
        
        if self.state == 0:
            assert False
        elif self.state == 1:
            self._reach1(info)
        elif self.state == 2:
            self._reach2()
        pass
    
    def _move(self, target, dis):
        target.x += dis[0]
        target.y += dis[1]
    
    def _trash(self, can_id):
        self.world.shelf.cans.pop(can_id)
    
    def _reach1(self, target):
        can_id, can = target
        dx, dy = can.x+40 - self.x, can.y - self.y
        kwargs = {"target": self, "dis": (0, np.sign(dy)*10)}
        self.action_holder.extend([(self._move, kwargs) for _ in range(abs(dy) // 10)])
        kwargs = {"target": self, "dis": (np.sign(dx)*10, 0)}
        self.action_holder.extend([(self._move, kwargs) for _ in range(abs(dx) // 10)])
        kwargs1 = {"target": self, "dis": (-np.sign(dx)*10, 0)}
        kwargs2 = {"target": self.world.shelf.cans[can_id], "dis": (-np.sign(dx)*10, 0)}
        for _ in range(abs(dx) // 10):
            self.action_holder.extend([(self._move, kwargs1), (self._move, kwargs2)])
        kwargs = {"can_id": can_id}
        self.action_holder.append((self._trash, kwargs))
        
    def _reach2(self):
        can_id, can = 0, self.world.shelf.get_cans()[0]
        dx, dy = can.x+40 - self.x, can.y - self.y
        kwargs = {"target": self, "dis": (0, np.sign(dy)*10)}
        self.action_holder.extend([(self._move, kwargs) for _ in range(abs(dy) // 10)])
        kwargs = {"target": self, "dis": (np.sign(dx)*10, 0)}
        self.action_holder.extend([(self._move, kwargs) for _ in range(abs(dx) // 10)])
        kwargs1 = {"target": self, "dis": (-np.sign(dx)*10, 0)}
        kwargs2 = {"target": self.world.shelf.cans[can_id], "dis": (-np.sign(dx)*10, 0)}
        for _ in range(abs(dx) // 10):
            self.action_holder.extend([(self._move, kwargs1), (self._move, kwargs2)])
        
    def step(self):
        if len(self.action_holder) > 0:
            func, kwargs = self.action_holder.pop(0)
            func(**kwargs)
        else:
            self._report()
            info = self._flow()
            self._run(info)
            

class World(OB):
    def __init__(self, name):
        super(World, self).__init__(name, world=None, size=(1000,1200), pos=(500,600))
        self.robot = Robot(name="robot", world=self)
        self.shelf = Shelf(name="shelf", world=self)
        self.layers = OrderedDict([
            (self.name, np.ones(list(self.size)+[4]).astype(np.uint8) * 255), 
            (self.shelf.name, np.ones(list(self.size)+[4]).astype(np.uint8) * 255),
            (self.robot.name, np.ones(list(self.size)+[4]).astype(np.uint8) * 255), 
        ])
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        self.render()
        
    def step(self):
        self.robot.step()
        self.shelf.step()
        self.render()
        
    
    def render(self):
        robot = self.robot
        shelf = self.shelf
        # TODO: 毎回呼ばなくて済む用にする
        self.layers[shelf.name] = layerize(self.size, (shelf.x, shelf.y), shelf.render())
        self.layers[robot.name] = layerize(self.size, (robot.x, robot.y), robot.render())
        layer = None
        for key, sub_layer in self.layers.items():
            if type(layer) == type(None):
                layer = sub_layer.copy()
            else:
                layer = overlay(layer, sub_layer.copy())
        cv2.imshow(self.name, layer[:,:,2:-5:-1]) # [2,1,0]
        cv2.waitKey(5)
        
        return layer
        
    def __del__(self):
        cv2.destroyWindow(self.name)

