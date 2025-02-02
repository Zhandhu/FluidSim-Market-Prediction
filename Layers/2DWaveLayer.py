import numpy as np
import torch 
import torch.nn as nn

class simLayer (nn.Module):
    def __init__ (self, input_size, dense_size, fluid_density = 1, fluid_viscosity = 0.1, poisson_iterations = 50, time_step_size = 0.1, iterations = 5):
        self.iterations = iterations 
        self.input_size = input_size
        self.u = nn.Parameter(torch.randn((input_size[0], input_size[1]+2, input_size[2]+2, dense_size)))
        self.v = nn.Parameter(torch.randn((input_size[0], input_size[1]+2, input_size[2]+2, dense_size)))
        self.rho = fluid_density
        self.nu = fluid_viscosity
        self.F = 1
        self.dt = time_step_size
        self.nit = poisson_iterations 
        self.dx = 2 / (input_size[0] + 1)
        self.dy = 2 / (input_size[1] + 1)
        self.preprocess = nn.Linear(input_size[-1], dense_size)

    def forward(self, x):
        p = nn.functional.pad(x, [0,0,1,1,1,1,0,0], mode = 'constant',value = 0)
        p = self.preprocess(p)
        for _ in range(self.iterations):
            b = torch.zeros_like(self.u)
            b[:, 1:-1, 1:-1,:] = (self.rho * (1 / self.dt * ((self.u[:, 1:-1, 2:,:] - self.u[:, 1:-1, 0:-2,:]) / (2 * self.dx) + 
                                                        (self.v[:, 2:, 1:-1,:] - self.v[:, 0:-2, 1:-1,:]) / (2 * self.dy)) -
                                                        ((self.u[:, 1:-1, 2:,:] - self.u[:, 1:-1, 0:-2,:]) / (2 * self.dx))**2 -
                                                        2 * ((self.u[:, 2:, 1:-1,:] - self.u[:, 0:-2, 1:-1,:]) / (2 * self.dy) * 
                                                                (self.v[:, 1:-1, 2:,:] - self.v[:, 1:-1, 0:-2,:]) / (2 * self.dx)) -
                                                        ((self.v[:, 2:, 1:-1,:] - self.v[:, 0:-2, 1:-1,:]) / (2 * self.dy))**2))
            
            # Periodic BC Pressure @ x = 2
            b[:, 1:-1, -1,:] = (self.rho * (1 / self.dt * ((self.u[:, 1:-1, 0,:] - self.u[:, 1:-1, -2,:]) / (2 * self.dx) + 
                                                        (self.v[:, 2:, -1,:] - self.v[:, 0:-2, -1,:]) / (2 * self.dy)) -
                                                        ((self.u[:, 1:-1, 0,:] - self.u[:, 1:-1, -2,:]) / (2 * self.dx))**2 -
                                                        2 * ((self.u[:, 2:, -1,:] - self.u[:, 0:-2, -1,:]) / (2 * self.dy) *
                                                            (self.v[:, 1:-1, 0,:] - self.v[:, 1:-1, -2,:]) / (2 * self.dx)) -
                                                        ((self.v[:, 2:, -1,:] - self.v[:, 0:-2, -1,:]) / (2 * self.dy))**2))

            # Periodic BC Pressure @ x = 0
            b[:, 1:-1, 0,:] = (self.rho * (1 / self.dt * ((self.u[:,1:-1, 1, :] - self.u[:,1:-1, -1, :]) / (2 * self.dx) +
                                                        (self.v[:,2:, 0, :] - self.v[:,0:-2, 0, :]) / (2 * self.dy)) -
                                                        ((self.u[:,1:-1, 1, :] - self.u[:,1:-1, -1, :]) / (2 * self.dx))**2 -
                                                        2 * ((self.u[:,2:, 0, :] - self.u[:,0:-2, 0, :]) / (2 * self.dy) *
                                                            (self.v[:,1:-1, 1, :] - self.v[:,1:-1, -1, :]) / (2 * self.dx))-
                                                        ((self.v[:,2:, 0, :] - self.v[:,0:-2, 0, :]) / (2 * self.dy))**2))

            pn = torch.empty_like(p)
            
            for q in range(self.nit):
                pn = p.clone()
                p[:, 1:-1, 1:-1,:] = (((pn[:,1:-1, 2:,:] + pn[:,1:-1, 0:-2,:]) * self.dy**2 + (pn[:,2:, 1:-1,:] + pn[:,0:-2, 1:-1,:]) * self.dx**2) / (2 * (self.dx**2 + self.dy**2)) -
                                    self.dx**2 * self.dy**2 / (2 * (self.dx**2 + self.dy**2)) * b[1:-1, 1:-1])

                # Periodic BC Pressure @ x = 2
                p[:,1:-1, -1,:] = (((pn[:,1:-1, 0,:] + pn[:,1:-1, -2,:])* self.dy**2 +
                                    (pn[:,2:, -1,:] + pn[:,0:-2, -1,:]) * self.dx**2) / (2 * (self.dx**2 + self.dy**2)) -
                                    self.dx**2 * self.dy**2 / (2 * (self.dx**2 + self.dy**2)) * b[1:-1, -1])

                # Periodic BC Pressure @ x = 0
                p[:,1:-1, 0,:] = (((pn[:,1:-1, 1,:] + pn[:,1:-1, -1,:])* self.dy**2 +
                                (pn[:,2:, 0,:] + pn[:,0:-2, 0,:]) * self.dx**2) / (2 * (self.dx**2 + self.dy**2)) -
                                self.dx**2 * self.dy**2 / (2 * (self.dx**2 + self.dy**2)) * b[:,1:-1, 0])
                
                # Wall boundary conditions, pressure
                p[:,-1, :,:] =p[:,-2, :,:]  # dp/dy = 0 at y = 2
                p[:,0, :,:] = p[:,1, :,:]  # dp/dy = 0 at y = 0

            un = self.u.clone()
            vn = self.v.clone()
            
            self.u[:, 1:-1, 1:-1] = (un[:, 1:-1, 1:-1] - un[:, 1:-1, 1:-1] * self.dt / self.dx * (un[:, 1:-1, 1:-1] - un[:, 1:-1, 0:-2]) -
                                                        vn[:, 1:-1, 1:-1] * self.dt / self.dy * (un[:, 1:-1, 1:-1] - un[:, 0:-2, 1:-1]) -
                                                        self.dt / (2 * self.rho * self.dx) * (p[:, 1:-1, 2:] - p[:, 1:-1, 0:-2]) + 
                                                        self.nu * (self.dt / self.dx**2 * (un[:, 1:-1, 2:] - 2 * un[:, 1:-1, 1:-1] + un[:, 1:-1, 0:-2]) + 
                                                                    self.dt / self.dy**2 * (un[:, 2:, 1:-1] - 2 * un[:, 1:-1, 1:-1] + un[:, 0:-2, 1:-1])) + 
                                                        self.F * self.dt)

            self.v[:, 1:-1, 1:-1] = (vn[:, 1:-1, 1:-1] - un[:, 1:-1, 1:-1] * self.dt / self.dx * (vn[:, 1:-1, 1:-1] - vn[:, 1:-1, 0:-2]) -
                                                        vn[:, 1:-1, 1:-1] * self.dt / self.dy * (vn[:, 1:-1, 1:-1] - vn[:, 0:-2, 1:-1]) -
                                                        self.dt / (2 * self.rho * self.dy) * (p[:, 2:, 1:-1] - p[:, 0:-2, 1:-1]) +
                                                        self.nu * (self.dt / self.dx**2 * (vn[:, 1:-1, 2:] - 2 * vn[:, 1:-1, 1:-1] + vn[:, 1:-1, 0:-2]) +
                                                                    self.dt / self.dy**2 * (vn[:, 2:, 1:-1] - 2 * vn[:, 1:-1, 1:-1] + vn[:, 0:-2, 1:-1])))

            # Periodic BC u @ x = 2     
            self.u[:, 1:-1, -1, :] = (un[:, 1:-1, -1, :] - un[:, 1:-1, -1, :] * self.dt / self.dx * (un[:, 1:-1, -1, :] - un[:, 1:-1, -2, :]) -
                                                        vn[:, 1:-1, -1, :] * self.dt / self.dy * (un[:, 1:-1, -1, :] - un[:, 0:-2, -1, :]) -
                                                        self.dt / (2 * self.rho * self.dx) * (p[:, 1:-1, 0, :] - p[:, 1:-1, -2, :]) + 
                                                        self.nu * (self.dt / self.dx**2 * (un[:, 1:-1, 0, :] - 2 * un[:, 1:-1,-1, :] + un[:, 1:-1, -2, :]) +
                                                                    self.dt / self.dy**2 * (un[:, 2:, -1, :] - 2 * un[:, 1:-1, -1, :] + un[:, 0:-2, -1, :])) + 
                                                        self.F * self.dt)

            # Periodic BC u @ x = 0
            self.u[:, 1:-1, 0, :] = (un[:, 1:-1, 0, :] - un[:, 1:-1, 0, :] * self.dt / self.dx * (un[:, 1:-1, 0, :] - un[:, 1:-1, -1, :]) -
                                                        vn[:, 1:-1, 0, :] * self.dt / self.dy * (un[:, 1:-1, 0, :] - un[:, 0:-2, 0, :]) - 
                                                        self.dt / (2 * self.rho * self.dx) * (p[:, 1:-1, 1, :] - p[:, 1:-1, -1, :]) + 
                                                        self.nu * (self.dt / self.dx**2 * (un[:, 1:-1, 1, :] - 2 * un[:, 1:-1, 0, :] + un[:, 1:-1, -1, :]) +
                                                                    self.dt / self.dy**2 * (un[:, 2:, 0, :] - 2 * un[:, 1:-1, 0, :] + un[:, 0:-2, 0, :])) + 
                                                        self.F * self.dt)

            # Periodic BC v @ x = 2
            self.v[:, 1:-1, -1, :] = (vn[:, 1:-1, -1, :] - un[:, 1:-1, -1, :] * self.dt / self.dx * (vn[:, 1:-1, -1, :] - vn[:, 1:-1, -2, :]) - 
                                                        vn[:, 1:-1, -1, :] * self.dt / self.dy * (vn[:, 1:-1, -1, :] - vn[:, 0:-2, -1, :]) -
                                                        self.dt / (2 * self.rho * self.dy) * (p[:, 2:, -1, :] - p[:, 0:-2, -1, :]) +
                                                        self.nu * (self.dt / self.dx**2 * (vn[:, 1:-1, 0, :] - 2 * vn[:, 1:-1, -1, :] + vn[:, 1:-1, -2, :]) +
                                                                    self.dt / self.dy**2 * (vn[:, 2:, -1, :] - 2 * vn[:, 1:-1, -1, :] + vn[:, 0:-2, -1, :])))

            # Periodic BC v @ x = 0
            self.v[:, 1:-1, 0, :] = (vn[:, 1:-1, 0, :] - un[:, 1:-1, 0, :] * self.dt / self.dx * (vn[:, 1:-1, 0, :] - vn[:, 1:-1, -1, :]) - 
                                                        vn[:, 1:-1, 0, :] * self.dt / self.dy * (vn[:, 1:-1, 0, :] - vn[:, 0:-2, 0, :]) - 
                                                        self.dt / (2 * self.rho * self.dy) * (p[:, 2:, 0, :] - p[:, 0:-2, 0, :]) +
                                                        self.nu * (self.dt / self.dx**2 * (vn[:, 1:-1, 1, :] - 2 * vn[:, 1:-1, 0, :] + vn[:, 1:-1, -1, :]) +
                                                                    self.dt / self.dy**2 * (vn[:, 2:, 0, :] - 2 * vn[:, 1:-1, 0, :] + vn[:, 0:-2, 0, :])))

                
            self.u[:, 0, :, :] = 0
            self.u[:, -1, :] = 0
            self.v[:, 0, :, :] = 0
            self.v[:, -1, :, :] = 0
        return torch.reshape(p[:,-1,:,:], (p.size(dim = 0), p.size(dim = 2), p.size(dim = 3)))