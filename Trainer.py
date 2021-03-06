# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 19:13:09 2021

@author: GKSch
"""

import torch
import torch.nn as nn
import NN
import numpy as np
import pickle
import os
import Vortex_Panel_Solver as vps
import matplotlib.pyplot as plt

class Trainer:
    
    def __init__(self, alpha, decay, n_panels, buffer_size, path=''):
        
        # Initialize model
        self.model = NN.NN(n_panels);
        self.n_panels = n_panels
        
        # Load model
        if len(path) != 0:
            self.load(path)
        
        # Initialize loss criterion, and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer_forward = torch.optim.Adam(self.model.parameters(), lr=alpha)
        self.optimizer_backward = torch.optim.Adam(self.model.parameters(), lr=alpha)
        self.lr_scheduler_forward = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_forward, gamma=decay)
        self.lr_scheduler_backward = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_backward, gamma=decay)
        
        # Initialize frame buffer
        self.buffer_size = buffer_size
        self.panels_forward_buffer = []
        self.performance_forward_buffer = []
        self.panels_backward_buffer = []
        self.performance_backward_buffer = []
        
    def load(self, path):
        
        with open(path, 'rb') as file:
            loaded_trainer = pickle.load(file)
        self.model.load_state_dict(loaded_trainer.state_dict())
 
    def forward(self, panels):
        
        # Convert frame to proper data type
        with torch.no_grad():
            panels = torch.tensor(panels).float()
            performance = self.model.forward(panels)
            
        # convert encoded frame to proper data type
        performance = performance.numpy()
        
        # Return the encoded frame of the proper data type
        return performance
       
    def backward(self, performance):
        
        # Convert frame to proper data type
        with torch.no_grad():
            performance = torch.tensor(performance).float()
            panels = self.model.backward(performance)
            
        # convert encoded frame to proper data type
        panels = panels.numpy()
        
        # Return the encoded frame of the proper data type
        return panels
    
    def update_forward(self, panels, performance):
        
        # Store the current panels and performance
        self.panels_forward_buffer.append(panels)
        self.performance_forward_buffer.append(performance)

        # If the buffer is full, perform one epoch of stochastic gradient descent
        if len(self.panels_forward_buffer) >= self.buffer_size:
            
            # Step through frame buffer
            tot_MSE_loss = 0.0
            
            for i in range(self.buffer_size):
                
                # Convert panels to proper data type
                with torch.no_grad():
                    panels = torch.tensor(self.panels_forward_buffer[i]).float()
                    target = torch.tensor(self.performance_forward_buffer[i]).float()
                    
                # Forward propogate the frame through the autoencoder
                performance_output = self.model.forward(panels)
            
                # Get loss
                curr_MSE_loss = self.criterion(performance_output, target)
                        
                # Calculate loss and take optimization step and learning rate step
                self.optimizer_forward.zero_grad()
                curr_MSE_loss.backward()
                self.optimizer_forward.step()
                self.lr_scheduler_forward.step()
                tot_MSE_loss = tot_MSE_loss + curr_MSE_loss.item()
            
            # Empty frame buffer
            self.panels_forward_buffer = []
            self.performance_forward_buffer = []

            return tot_MSE_loss / self.buffer_size
        
        return -1.0
    
    def update_backward(self, performance, panels):
        
        # Store the current panels and performance
        self.panels_backward_buffer.append(panels)
        self.performance_backward_buffer.append(performance)

        # If the buffer is full, perform one epoch of stochastic gradient descent
        if len(self.panels_backward_buffer) >= self.buffer_size:
            
            # Step through frame buffer
            tot_MSE_loss = 0.0
            
            for i in range(self.buffer_size):
                
                # Convert panels to proper data type
                with torch.no_grad():
                    performance = torch.tensor(self.performance_backward_buffer[i]).float()
                    target = torch.tensor(self.panels_backward_buffer[i]).float()
                    
                # Forward propogate the frame through the autoencoder
                panels_output = self.model.backward(performance)
            
                # Get loss
                curr_MSE_loss = self.criterion(panels_output, target)
                        
                # Calculate loss and take optimization step and learning rate step
                self.optimizer_backward.zero_grad()
                curr_MSE_loss.backward()
                self.optimizer_backward.step()
                self.lr_scheduler_backward.step()
                tot_MSE_loss = tot_MSE_loss + curr_MSE_loss.item()
                    
            # Empty frame buffer
            self.panels_backward_buffer = []
            self.performance_backward_buffer = []

            return tot_MSE_loss / self.buffer_size
        
        return -1.0
    
    def draw(self, path, forward_loss=[], backward_loss=[]):
        
        if len(forward_loss) > 0:
            forward_path = path+"/foward_learning.png"
            
            plt.close()
            
            plt.semilogy(np.arange(len(forward_loss)), forward_loss, color = 'r', lw = 3.0)
            
            plt.xlabel("Optimization Epoch [-]", fontsize="x-large")
            plt.ylabel("MSE Loss [-]", fontsize="x-large")
            plt.title("Foward Learning Curves (Estimate Performance)", fontsize="xx-large")
            
            plt.xticks(fontsize='x-large')
            plt.yticks(fontsize='x-large')
            
            plt.gcf().set_size_inches(8,5.6)
            plt.savefig(forward_path, dpi=200)
            
        if len(backward_loss) > 0:
            backward_path = path + "/backward_learning.png"
            
            plt.close()
            
            plt.semilogy(np.arange(len(backward_loss)), backward_loss, color = 'r', lw = 3.0)
            
            plt.xlabel("Optimization Epoch [-]", fontsize="x-large")
            plt.ylabel("MSE Loss [-]", fontsize="x-large")
            plt.title("Backward Learning Curves (Rebuild Airfoil)", fontsize="xx-large")
            
            plt.xticks(fontsize='x-large')
            plt.yticks(fontsize='x-large')
            
            plt.gcf().set_size_inches(8,5.6)
            plt.savefig(backward_path, dpi=200)
            
    def save(self, path):
        
        with open(path+"/trainer", 'wb') as file:
            pickle.dump(self.model, file)
            
    def save_results(self, path, n_airfoils, forward_loss=[], backward_loss=[]):

        print("\nSaving...")
        num = 0
        done = False
        while not done:
            if not (os.path.exists(path + "/set_" + str(num))):
                path = path + "/set_" + str(num)
                os.mkdir(path)
                done = True
            else:
                num = num + 1        
                
        self.draw(path, forward_loss=forward_loss, backward_loss=backward_loss)
        self.save(path)

        solver = vps.Solver()
        
        print("Plotting...")
        for i in range(n_airfoils):
            panels = vps.Panels(n_panels)
            panels.draw(path, airfoil_name='Airfoil_'+str(i))
            
            estimated_performance = self.forward(100.0 * panels.y_coords)
            estimated_performance[0] = estimated_performance[0] / 1.458
            estimated_performance[1] = estimated_performance[1] / 2.255
            estimated_performance[2] = estimated_performance[2] / 72.71
            estimated_performance[3] = estimated_performance[3] / 92.17
            estimated_performance[4] = estimated_performance[4] / 52.12
            estimated_performance[5] = estimated_performance[5] / 67.58
            estimated_performance[6] = estimated_performance[6] / 31.70
        
            performance_input, _, _, _ = solver.get_curves(panels, 5)
            performance_input_trunc = np.zeros(5)
            performance_input_trunc[0] = 1.458 * performance_input[0]
            performance_input_trunc[1] = 2.255 * performance_input[1]
            performance_input_trunc[2] = 72.71 * performance_input[2]
            performance_input_trunc[3] = 67.58 * performance_input[5]
            performance_input_trunc[4] = 31.70 * performance_input[6]
            rebuilt_panels_y_coords = self.backward(performance_input_trunc) / 118.5
            
            rebuilt_panels = vps.Panels(n_panels)
            rebuilt_panels.set_y_coords(rebuilt_panels_y_coords)
            rebuilt_panels.draw(path, airfoil_name='Rebuilt Airfoil_'+str(i))
            
            solver.draw_curves(path, panels, name='Airfoil_'+str(i), estimated_performance=estimated_performance, rebuilt_panels=rebuilt_panels)
        
if __name__ == '__main__':  
    
    n_panels = 48
    num_airfoils = 250000
    buffer_size = 100
    lr_start = 1.0e-3
    lr_end = 1.0e-5
    
    trainer = Trainer(lr_start, (lr_end/lr_start)**(1.0/num_airfoils), n_panels, buffer_size)
    solver = vps.Solver()
    
    forward_loss = []
    backward_loss = []
    for i in range(num_airfoils):
        if len(forward_loss) > 0:
            if forward_loss[-1] // 10.0 >= 1.0:
                forward_str = '{:.2f}'.format(round(forward_loss[-1],2))
            else:
                forward_str = '{:.3f}'.format(round(forward_loss[-1],3))
            if backward_loss[-1] // 10.0 >= 1.0:
                backward_str = '{:.2f}'.format(round(backward_loss[-1],2))
            else:
                backward_str = '{:.3f}'.format(round(backward_loss[-1],3))
            print("Airfoil: " + str(i+1) + "/" + str(num_airfoils) + "  |  Forward Loss: " + forward_str + "  |  Backward Loss: " + backward_str, end='\r')
        else:
            print("Airfoil: " + str(i+1) + "/" + str(num_airfoils) + "  |  Forward Loss: " + '{:.3f}'.format(round(0.0,3)) + "  |  Backward Loss: " + '{:.3f}'.format(round(0.0,3)), end='\r')
            
        panels = vps.Panels(n_panels)
        performance_input, _, _, _ = solver.get_curves(panels, 5)
        
        performance_input[0] = 1.458 * performance_input[0]
        performance_input[1] = 2.255 * performance_input[1]
        performance_input[2] = 72.71 * performance_input[2]
        performance_input[3] = 92.17 * performance_input[3]
        performance_input[4] = 52.12 * performance_input[4]
        performance_input[5] = 67.58 * performance_input[5]
        performance_input[6] = 31.70 * performance_input[6]
        
        performance_input_trunc = np.zeros(5)
        performance_input_trunc[0] = performance_input[0]
        performance_input_trunc[1] = performance_input[1]
        performance_input_trunc[2] = performance_input[2]
        performance_input_trunc[3] = performance_input[5]
        performance_input_trunc[4] = performance_input[6]
        
        curr_forward_loss = trainer.update_forward(118.5*panels.y_coords, performance_input)
        curr_backward_loss = trainer.update_backward(performance_input_trunc, 118.5*panels.y_coords)
        
        if(curr_forward_loss != -1.0):
            forward_loss.append(curr_forward_loss)
        if(curr_backward_loss != -1.0):
            backward_loss.append(curr_backward_loss)
           
    trainer.save_results("results", 10, forward_loss=forward_loss, backward_loss=backward_loss)
    print("Done!")