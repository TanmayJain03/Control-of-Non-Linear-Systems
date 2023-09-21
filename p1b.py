import  numpy  as  np 
import  matplotlib.pyplot  as  plt 
class  manipulator  : 
    def  __init__(self  ,  p_1  ,  p_3  ,  q_0  ,  q_r): 
    # system states 
        self.q = q_0 
        self.q_r = q_r 
        self.e = self.q - self.q_r 
        self.q_dot = 0   
        self.q_dot_dot = 0  
        self.xi = 0  
        self.xi_dot = 0 
        # constant values 
        self.p_1 = p_1 
        self.p_3 = p_3 
        self.dt = 0.1 
        self.update_mc  () 
    def  update_mc(self): 
        self.m = self.p_1  + (2  *  self.p_3  *  np.cos(self.q)) 
        self.c  = -  self.p_3  *  np.sin(self.q) *  self.q_dot 
        self.m_dot  = -  2  *  self.p_3  *  np.sin(self.q)  *  self.q_dot 
        self.c_dot  = -(self.p_3  *  np.cos(self.q) * (self.q_dot  **  2)) - (self.p_3  * np.sin(self.q) *  self.q_dot_dot) 
    def  advance_state(self  ,  tau): 
        dt = self.dt 
        # dynamics 
        self.q_dot_dot  = (self.xi  - (self.c  *  self.q_dot))  /  self.m 
        self.xi_dot = tau 
        # update states 
        self.q_dot  +=  self.q_dot_dot  *  dt 
        self.q  +=  self.q_dot  *  dt 
        self.xi  +=  self.xi_dot  *  dt 
        self.update_mc  () 
    def  compute_control_law(self):  # taking the control  lyapunov function to be V(q,q_dot) = e^2/2 + q_dot^2/2 
        # k_0 = Cq_dot + mq_r - mq - mq_dot 
        k_0_dot  = (self.c_dot  *  self.q_dot  +  self.c  *  self.q_dot_dot) +  self.m_dot * self.q_r  - (self.m_dot  *  self.q  +  self.m  *  self.q_dot)  - (self.m_dot  *  self.q_dot  + self.m  *  self.q_dot_dot) 
        self.tau = k_0_dot  - (2  *  self.q_dot  /  self.m)  - (self.xi  /(self.m  **  2)) + (self.c  *  self.q_dot  /(self.m  **  2)) + ((self.q_r - self.q)/  self.m) 
    def  simulate(self  ,  t_f) : 
        self.t = np.linspace(0  ,  t_f  ,  int(t_f  /  self.dt)) 
        self.compute_control_law  () 
        self.tau_array  = [] 
        self.q_array  = [] 
        self.q_r_array  = [] 
        for  _  in  range(len(self.t)): 
            self.tau_array.append(self.tau) 
            self.q_array.append(self.q) 
            self.q_r_array.append(self.q_r) 
            self.advance_state(self.tau) 
            self.compute_control_law  () 
    def  plot(self): 
        fig  ,  ax = plt.subplots(1  ,  2) 
        ax  [  0  ].  plot(self.t  ,  self.tau_array  ,  '-g') 
        # ax[0].plot(self.t, self.tau_array, '-r') 
        ax  [  0  ].  legend  ([  "tau"  ]) 
        ax  [  0  ].  set_ylabel("tau") 
        ax  [  0  ].  set_xlabel("t") 
        ax  [  0  ].  grid  () 
        ax  [  0  ].  set_title("Control input tau v/s iteration  t") 
        ax  [  1  ].  plot(self.t  ,  self.q_array  ,  '-g') 
        ax  [  1  ].  plot(self.t  ,  self.q_r_array  ,  ':b') 
        ax  [  1  ].  legend  ([  "q"  ,  "q_ref"  ]) 
        ax  [  1  ].  set_ylabel("q") 
        ax  [  1  ].  set_xlabel("t") 
        ax  [  1  ].  grid  () 
        ax  [  1  ].  set_title("Control input q v/s iteration  t") 
        plt.show  () 
if  __name__  ==  '__main__'  : 
    p_1 = 3.31 
    p_3 = 0.16 
    q_r = 0.3 
    q_0 = 1
    m = manipulator(p_1  ,  p_3  ,  q_0  ,  q_r) 
    m.simulate(120) 
    m.plot  () 