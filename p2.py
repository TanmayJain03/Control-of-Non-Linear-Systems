import  numpy  as  np 
import  matplotlib.pyplot  as  plt 
from  math  import  sqrt, tanh, tan 
class  rigid_body  : 
    def  __init__(self, J, rho_0, omega_0): 
        self.J  =  J 
        self.rho  =  rho_0 
        print(self.rho) 
        self.rho_cross  =  self.get_curl(self.rho) 
        self.omega  =  omega_0 
        self.I  =  np.identity(3) 
        self.dt  =  0.05 
        self.k_1  =  1  # storage function gain 
        self.k_2  =  12  # control law gain 
        self.compute_dW_drho  () 
    def  get_curl(self, vec): 
        curl  =  np.array  ([[  0  , -  vec[2],  vec[1] ], 
                             [vec[2] ,  0  , -  vec[0]], 
                             [-  vec[1],  vec[0] ,  0  ]]) 
        return  curl 
    # now compute the jacobian of the storage function to compute the control law 
    def  compute_dW_drho(self): 
        self.dW_drho  = (  2  *  self.k_1  / (  1  +  np.dot(self.rho  .T,  self.rho))) *  self.rho 
    
    def  advance_state(self, u) : 
        dt  =  self.dt 
        # dynamics 
        self.rho_dot  =  np.dot  ((  self.I  +  self.rho_cross  +  np.dot(self.rho  , self.rho  .T)),  self.omega) 
        self.omega_dot  =  np.dot(np.linalg.inv(self.J),(-(  np.cross(self.omega  , np.dot(self.J, self.omega))) +  u)) 
        self.y  =  self.omega 
        self.compute_dW_drho  () 
        # update states 
        self.omega  +=  self.omega_dot  *  dt 
        self.rho  +=  self.rho_dot  *  dt 
        self.rho_cross  =  self.get_curl(self.rho) 
    def  compute_control_law(self): 
        self.nu  = -  self.k_2  *  np.tanh(self.omega) 
        self.u  =  self.nu  -  np.dot(self.dW_drho  ,(  self.I  +  self.rho_cross  + 
        np.dot(self.rho, self.rho  .T))).T 
    def  simulate(self, t_f): 
        self.t  =  np.linspace(0, t_f, int(t_f  /  self.dt)) 
        self.compute_control_law  () 
        self.u_1_array  = [] 
        self.u_2_array  = [] 
        self.u_3_array  = [] 
        self.rho_1_array  = [] 
        self.rho_2_array  = [] 
        self.rho_3_array  = [] 
        self.omega_1_array  = [] 
        self.omega_2_array  = [] 
        self.omega_3_array  = [] 
        for  _  in  range(len(self.t)): 
            self.rho_1_array.append(self.rho[0]) 
            self.rho_2_array.append(self.rho[1]) 
            self.rho_3_array.append(self.rho[2]) 
            self.omega_1_array.append(self.omega[0]) 
            self.omega_2_array.append(self.omega[1]) 
            self.omega_3_array.append(self.omega[2]) 
            self.u_1_array.append(self.u[0]) 
            self.u_2_array.append(self.u[1]) 
            self.u_3_array.append(self.u[2]) 
            self.advance_state(self.u) 
            self.compute_control_law  () 
    def  plot(self): 
        fig, ax  =  plt.subplots(1, 3) 
        ax[0].plot(self.t, self.rho_1_array, '-b') 
        ax[0].plot(self.t, self.rho_2_array, '-g') 
        ax[0].plot(self.t, self.rho_3_array, '-r') 
        ax[0].legend  ([  "rho1", "rho2", "rho3"  ]) 
        ax[0].set_ylabel("v") 
        ax[0].set_xlabel("t") 
        ax[0].grid  () 
        ax[0].set_title("Components of Rho vs iteration t") 
        ax[1].plot(self.t, self.omega_1_array, '-b') 
        ax[1].plot(self.t, self.omega_2_array, '-g') 
        ax[1].plot(self.t, self.omega_3_array, '-r') 
        ax[1].legend  ([  "omega1", "omega2", "omega3"  ]) 
        ax[1].set_ylabel("w") 
        ax[1].set_xlabel("t") 
        ax[1].grid  () 
        ax[1].set_title("Components of Omega vs iteration t") 
        ax[2].plot(self.t, self.u_1_array, '-b') 
        ax[2].plot(self.t, self.u_2_array, '-g') 
        ax[2].plot(self.t, self.u_3_array, '-r') 
        ax[2].legend  ([  "u1", "u2", "u3"  ]) 
        ax[2].set_ylabel("w") 
        ax[2].set_xlabel("t") 
        ax[2].grid  () 
        ax[2].set_title("Components of input u vs iteration t") 
        plt.show  () 
if  __name__  ==  '__main__'  : 
    alpha_1  =  0 
    theta_initial  = -  35 
    J  =  np.array  ([[  20, 1.2, 0.9  ], [1, 17, 1.4  ], [0, 1.4, 15  ]]) 
    orient  =  np.array  ([  1  /  sqrt(3),  1  /  sqrt(3),  1  /  sqrt(3)]).T 
    k  =  np.array  ([  1, 0, 0  ]).T 
    rho_0  =  orient  *  np.cos(theta_initial) +  np.cross(k  , 
    orient)*  np.sin(theta_initial) +  k  *(  np.dot(k, orient))*(  1  -  np.cos(theta_initial)) 
    omega_0  =  np.array  ([  0.0, 0.0, 0.0  ]).T 
    rot_dyn  =  rigid_body(J, rho_0, omega_0) 
    rot_dyn.simulate(35) 
    rot_dyn.plot()