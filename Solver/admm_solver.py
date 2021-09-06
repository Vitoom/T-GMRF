#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 21:43:06 2019

@author: vito
"""

import numpy
import math

class ADMMSolver:
    def __init__(self, ic_sequence, ic_index, lamb, beta, variables_dim, windows_dim, rho, S, rho_update_func=None):
        self.ic_sequence = ic_sequence
        self.ic_index = ic_index
        self.lamb = lamb
        self.beta = beta
        self.variables_dim = variables_dim
        self.windows_dim = windows_dim
        self.rho = rho
        self.S = S
        self.upper_triangle_size = int(self.variables_dim * (self.variables_dim + 1) / 2)
        self.x = numpy.zeros(self.upper_triangle_size)
        self.z = numpy.zeros(self.upper_triangle_size)
        self.u = numpy.zeros(self.upper_triangle_size)
        self.rho = float(rho)
        self.rho_update_func = rho_update_func
    
    def ij2symmetric(self, i, j, size):
        return (size * (size + 1))/2 - (size-i)*((size - i + 1))/2 + j - i
    
    def upper2Full(self, a):
        n = int((-1  + numpy.sqrt(1+ 8*a.shape[0]))/2)  
        A = numpy.zeros([n,n])
        A[numpy.triu_indices(n)] = a 
        temp = A.diagonal()
        A = (A + A.T) - numpy.diag(temp)             
        return A
    
    def Prox_logdet(self, S, A, eta):
        d, q = numpy.linalg.eigh(eta*A-S)
        q = numpy.matrix(q)
        X_var = ( 1/(2*float(eta)) )*q*( numpy.diag(d + numpy.sqrt(numpy.square(d) + (4*eta)*numpy.ones(d.shape))) )*q.T
        x_var = X_var[numpy.triu_indices(S.shape[1])] # extract upper triangular part as update variable      
        return numpy.matrix(x_var).T

    def ADMM_x(self):    
        a = self.z-self.u
        A = self.upper2Full(a)
        eta = self.rho
        x_update = self.Prox_logdet(self.S, A, eta)
        self.x = numpy.array(x_update).T.reshape(-1)
        
    def ADMM_z(self, index_penalty = 1):
        a = self.x + self.u
        b = numpy.zeros_like(a)
        coef = [1, -4, -4, 1]
        stride = [2, 1, -1, -2]
        for j in range(4):
            if self.ic_index + stride[j] >= 0 and self.ic_index + stride[j] < self.windows_dim:
                b += self.ic_sequence[self.ic_index + stride[j], :] * coef[j]
        probSize = self.variables_dim
        z_update = numpy.zeros(self.upper_triangle_size)

        for j in range(self.variables_dim):
            startPoint = j
            for k in range(startPoint, self.variables_dim):
                locList = [(j, k)]
                lamSum = sum(self.lamb[loc2, loc1] for (loc1, loc2) in locList)
                indices = [self.ij2symmetric(loc2, loc1, probSize) for (loc1, loc2) in locList]
                pointSum = sum(a[int(index)] for index in indices)
                rhoPointSum = self.rho * pointSum
                delaySum = sum(self.beta[loc2, loc1] * b[int(self.ij2symmetric(loc2, loc1, probSize))]  for (loc1, loc2) in locList)
                
                #Calculate soft threshold
                ans = 0
                #If answer is positive
                if rhoPointSum - delaySum > lamSum:
                    ans = max((rhoPointSum - delaySum - lamSum)/(self.rho + 6), 0)
                elif rhoPointSum - delaySum < lamSum:
                    ans = max((rhoPointSum + delaySum - lamSum)/(self.rho + 6), 0)
                    
                for index in indices:
                    z_update[int(index)] = ans
            self.z = z_update
    
    def ADMM_u(self):
        u_update = self.u + self.x - self.z
        self.u = u_update
    
    # Returns True if convergence criteria have been satisfied
    # eps_abs = eps_rel = 0.01
    # r = x - z
    # s = rho * (z - z_old)
    # e_pri = sqrt(length) * e_abs + e_rel * max(||x||, ||z||)
    # e_dual = sqrt(length) * e_abs + e_rel * ||rho * u||
    # Should stop if (||r|| <= e_pri) and (||s|| <= e_dual)
    # Returns (boolean shouldStop, primal residual value, primal threshold,
    #          dual residual value, dual threshold)
    def CheckConvergence(self, z_old, e_abs, e_rel, verbose):
        norm = numpy.linalg.norm
        r = self.x - self.z
        s = self.rho * (self.z - z_old)
        # Primal and dual thresholds. Add .0001 to prevent the case of 0.
        e_pri = math.sqrt(self.upper_triangle_size) * e_abs + e_rel * max(norm(self.x), norm(self.z)) + .0001
        e_dual = math.sqrt(self.upper_triangle_size) * e_abs + e_rel * norm(self.rho * self.u) + .0001
        # Primal and dual residuals
        res_pri = norm(r)
        res_dual = norm(s)
        if verbose:
            # Check Convergence
            print('  r:', res_pri)
            print('  e_pri:', e_pri)
            print('  s:', res_dual)
            print('  e_dual:', e_dual)
        stop = (res_pri <= e_pri) and (res_dual <= e_dual)
        return (stop, res_pri, e_pri, res_dual, e_dual)

    def __call__(self, maxIters, eps_abs, eps_rel, verbose):
        self.status = 'Incomplete: max iterations reached'
        for i in range(maxIters):
            z_old = numpy.copy(self.z)
            self.ADMM_x()
            self.ADMM_z()
            self.ADMM_u()
            if i != 0:
                stop, res_pri, e_pri, res_dual, e_dual = self.CheckConvergence(z_old, eps_abs, eps_rel, verbose)
                if stop:
                    # early stop if converged
                    self.status = 'Optimal'
                    break
                new_rho = self.rho
                if self.rho_update_func:
                    new_rho = self.rho_update_func(self.rho, res_pri, e_pri, res_dual, e_dual)
                scale = self.rho / new_rho
                self.rho = new_rho
                self.u = scale * self.u
            if verbose:
                print('Iteration %d' % i)
        return self.x
