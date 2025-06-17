import os
import random
import math
from typing import List, Optional
from PIL import Image

class HopfieldNetwork:

    def __init__(self, dims: List[int]) -> None:  
        self.dims = dims
        self.num_nodes = dims[0] * dims[1]
        self.network_wts = [[0 for _ in range(self.num_nodes)] for _ in range(self.num_nodes)]

    def read_pbm_imgs(self, train_img_dir: str) -> List[str]:   # reading .pbm files into vectors
        pbm_images = []                                        
        for pbm_img in os.listdir(train_img_dir):               # maintaining a list of the names of the images in the dir
            pbm_images.append(self.pbm_to_vec(train_img_dir + os.fsdecode(pbm_img)))
        return pbm_images
    
    def vec_to_png(self, vec: List[int], fname: str) -> None:   # saving the pbm in vector form to a PNG file
        img = []
        for i in range(self.dims[0]):
            r = []
            for j in range(self.dims[1]):
                if vec[i * self.dims[1] + j] == 1:
                    r.append(0)
                else:
                    r.append(255)
            img.append(r)
            
        pixels = []
        for row in img:
            for pixel in row:
                pixels.append(pixel)
                
        img_png = Image.new('L',(self.dims[0], self.dims[1]))
        img_png.putdata(pixels)
        img_png.save(fname)

    def pbm_to_vec(self, img_path: str) -> list[int]:           # storing the pixels of a .pbm file in the form of a vector
        f = open(img_path);
        i = 0;
        c = 0;
        for line in f:
            line = line.strip().split(" ")
            if(i == 2):
                dim1 = int(line[0])
                dim2 = int(line[1])
                pbm_vec = [0 for _ in range(dim1 * dim2)]
            if(i > 2):
                for j in range(len(line)):
                    if(line[j] == "1"):
                        pbm_vec[c] = 1
                    elif(line[j] == "0"):
                        pbm_vec[c] = -1
                    c += 1
            i += 1
        return pbm_vec
    
    def viz_pbm_vec(self, pbm_vec: list) -> None:               # visulaizing the image (in vector form) 
        for i in range(32): print("━", end='')
        print()

        i = 0
        for i in range(len(pbm_vec)):
            if pbm_vec[i] == 1:
                print("██", end='')
            elif pbm_vec[i] == -1:
                print("  ", end='')
            if(i + 1) % 16 == 0:
                print() 

        for i in range(32): print("━", end='')
        print()
        
    def corrupt_pbm_vec(self, pbm_vec : list, prob : float) -> list[int]:   # looping over each pixel and flipping it with 'prob' probability
        corrupted_vec = pbm_vec.copy()
        for i in range(0, len(pbm_vec)):
            r = random.random()
            if(r < prob):
                if(pbm_vec[i] == 1): 
                    corrupted_vec[i] = -1;
                else:
                    corrupted_vec[i] = 1;

        return corrupted_vec
    
    def crop_pbm_vec(self, pbm_vec : list, crop_side : int, tl : list, w : bool) -> list[int]:  # cropping the image by retaining only (crop_side x crop_side)
                                                                                                # area and blackening/whitening out the rest of the image
                                                                                                # tl defines the location o the cropping square
        cropped_vec = pbm_vec.copy()
        row = 0
        col = 0
        for i in range(0, len(pbm_vec)):
            
            if(not((row >= tl[0]) and (row < (crop_side+tl[0])) and (col >= tl[1]) and (col < (crop_side+tl[1])))):
                cropped_vec[i] = -1 if w else 1
                
            col += 1
            if(col == math.sqrt(len(pbm_vec))):
                col = 0
                row += 1
                
        return cropped_vec
    
    def get_similarity(self, vec1 : list, vec2 : list) -> float:    # a metric to measure what fraction of pixels are the same in two images
        num = 0
        den = len(vec1)
        for i in range(len(vec1)):
            if(vec1[i]==vec2[i]):
                num += 1
                
        return num/den

    def compare_vecs(self, vec1: list, vec2: list) -> bool:         # checking if two vectors are the same
        if(len(vec1) == len(vec2)):
            for i in range(len(vec1)):
                if(vec1[i] != vec2[i]):
                    return False
        else:
            return False
        
        return True
    
    def hebbian_train(self, input_imgs_dir: str) -> None:           # training the Hopfield Network
        
        input_imgs = self.read_pbm_imgs(input_imgs_dir)
        for input_img in input_imgs:                                # ensuring dimensional compatibility
            if(len(input_img) != self.num_nodes):
                raise ValueError("The input img is not of the network's dimension")

        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if(i == j):
                    self.network_wts[i][j] = 0                      # ensuring that all 'self loop' edges are 0
                else:
                    node_pdt_sum = 0
                    for k in range(0, len(input_imgs)):
                        node_pdt_sum += (input_imgs[k][i] * input_imgs[k][j])   # building the sum to update the network's weight
                    self.network_wts[i][j] = (node_pdt_sum / len(input_imgs))   # update rule
    
    def inference_sync(self, input_img: list, save_path: Optional[str] = None, visualise: bool = False) -> List[int]:
                                                                # obtaining the network's output in synchronous mode
        prev_state = []
        current_state = input_img.copy()

        if save_path:
            os.makedirs(save_path)
            self.vec_to_png(current_state, f"{save_path}/input.png")
        if visualise:
            print("Input:")
            self.viz_pbm_vec(current_state)

        c = 1       # iteration counter
        
        while True:
            
            prev_state = current_state.copy()
            wted_sum = 0

            i = 0
            for i in range(len(self.network_wts)):
                
                j = 0
                wted_sum = 0
                for j in range(len(self.network_wts[i])):
                    wted_sum += self.network_wts[i][j] * prev_state[j]  # calculating sum to decide the corresponding pixel's update based on previous state
                
                if(wted_sum >= 0):     # update rule
                    current_state[i] = 1
                else:
                    current_state[i] = -1
                    
            if self.compare_vecs(prev_state, current_state):    # checking if inference has been completed
                break
            
            if save_path:
                self.vec_to_png(current_state, f"{save_path}/iter_{c}.png") # saving this iteration's result
            if visualise:
                print(f"State {c}")
                self.viz_pbm_vec(current_state)                             # visualizing the current state

            c += 1
            
        return current_state 
    
    def inference_async(self, input_img: list, save_path: Optional[str] = None, visualise: bool = False) -> List[int]:
                                                                # obtaining the network's output in asynchronous mode
        prev_state = []
        current_state = input_img.copy()

        if save_path:
            os.makedirs(save_path)
            self.vec_to_png(current_state, f"{save_path}/input.png")
        if visualise:
            print("Input:")
            self.viz_pbm_vec(current_state)

        c = 1
        
        while not self.compare_vecs(prev_state, current_state):
            
            prev_state = current_state.copy()
            wted_sum = 0

            indices = list(range(len(self.network_wts)))
            random.shuffle(indices)
            z = 0
            for i in indices:
                wted_sum = 0
                for j in range(len(self.network_wts[i])):
                    wted_sum += self.network_wts[i][j] * current_state[j]   # calculating sum to decide the corresponding pixel's update based on current state
                if(wted_sum >= 0):
                    current_state[i] = 1
                else:
                    current_state[i] = -1
                z += 1

                if(z % 100 == 0 and save_path):
                    self.vec_to_png(current_state, f"{save_path}/iter_{c}_pixel_{z}.png")
            
            if self.compare_vecs(prev_state, current_state):                # checking if inference has been completed
                break
            
            if(save_path):
                self.vec_to_png(current_state, f"{save_path}/iter_{c}_final.png") # saving this iteration's result
            if visualise:
                print(f"State {c}")
                self.viz_pbm_vec(current_state)                                   # visualizing the current state
            
            c += 1
            
        return current_state 