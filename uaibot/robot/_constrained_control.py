from uaibot.utils import Utils
import numpy as np

def funF(r):
    f = np.matrix(r)
    for j in range(np.shape(r)[0]):
        f[j,0] = np.sign(r[j,0])*np.sqrt(np.abs(r[j,0]))
        
    return f

def _constrained_control(self, htm_tg, q=None, obstacles=[], htm=None, 
                              Kp =  2.0, eta_obs = 0.3, eta_auto = 0.3, eta_joint = 0.3, 
                              eps_to_obs = 0.003, h_to_obs = 0.003, 
                              eps_auto = 0.02,  h_auto = 0.05, 
                              d_safe_obs = 0.02, d_safe_auto = 0.002, d_safe_jl = (np.pi/180)*5,
                              eps_reg = 0.01):

    n = len(self.links)
    
    if str(type(obstacles))=="<class 'list'>":
        obstacles_list = obstacles
    else:
        obstacles_list = [obstacles]

    if q is None:
        q = self.q
    if htm is None:
        htm = self.htm

    q_min = self.joint_limit[:,0]
    q_max = self.joint_limit[:,1]
    
    #Get smooth distance structure from robot to point cloud
    A_obj = np.matrix(np.zeros((0,n)))
    b_obj_raw =np.matrix(np.zeros((0,1)))
    

    
    for obj in obstacles_list:    
        dr_obj =self.compute_dist(obj = obj, q = q, eps=eps_to_obs, h=h_to_obs)
        A_obj = np.matrix(np.vstack(  (A_obj, dr_obj.jac_dist_mat)  ))
        b_obj_raw = np.matrix(np.vstack(  (b_obj_raw, dr_obj.dist_vect)  ))

        
        
    b_obj = -eta_obs * (b_obj_raw-d_safe_obs)
    
    #Get smooth distance structure from robot to itself (autocollision)
    dr_auto = self.compute_dist_auto(q = q, eps=eps_auto, h=h_auto)    

    A_auto = dr_auto.jac_dist_mat
    b_auto_raw = dr_auto.dist_vect
    
    b_auto = -eta_auto * (b_auto_raw-d_safe_auto)
    
    #Assemble matrices for joint limit constraints
        
    A_joint = np.matrix(np.vstack(  (np.identity(7), -np.identity(7))  ))
    b_joint_raw = np.matrix(np.vstack(  (q-q_min , q_max - q) )) 
    
    b_joint = -eta_joint * (b_joint_raw-d_safe_jl)
    
    #Get task function data
    
    r, Jr = self.task_function(htm_tg, q=q)
    
    #Create the optimization problem
    A =    np.matrix(np.vstack( (A_obj, A_auto, A_joint) ) )
    b =    np.matrix(np.vstack( (b_obj, b_auto, b_joint) ) )
    H = 2*(Jr.transpose() * Jr + eps_reg *np.identity(7))
    f = Jr.transpose() * Kp * funF(r)
    
    #Solve QP and obtain joint velocity
    qdot = Utils.solve_qp(H, f, A, b)
    
    #Compute some statistics
    ep = np.linalg.norm(r[0:3,0])
    eox = np.arccos(1-r[3,0])
    eoy = np.arccos(1-r[4,0])
    eoz = np.arccos(1-r[5,0])
    eo = (180/np.pi)* max([eox,eoy,eoz])
    
    dobs = b_obj_raw.min()
    dauto = b_auto_raw.min()
    djoint = b_joint_raw.min()
    
 
    return qdot, ep, eo, dobs, dauto, djoint
