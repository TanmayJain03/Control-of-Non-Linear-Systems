import  numpy  as  np 
from  math  import  sin ,  cos ,  tanh 
from  numpy . linalg  import  inv 
import  matplotlib . pyplot  as  plt 
# import scipy.integrate as integrate 
import  matplotlib . animation  as  animation 
class  RoboticManipulator (): 
    def  __init__ ( self , 
    q1_ref_0  =  0  , 
        q2_ref_0  =  0  , 
        q1_0  
        =  0  , 
        q2_0  
        =  0  , 
        q1dot_0  =  0  , 
        q2dot_0  =  0  , 
        q1dot2_0  =  0  , 
        q2dot2_0  =  0  ): 
        ### CONSTANTS 
        self.p1  =  3.31  
        self.p2  =  0.116 
        self.p3  =  0.16 
        self.L1  =  0.8  
        #kg.m^2 
        #m 
        self.L2  =  0.5 
        self.O1  = - 0.666  #m 
        self.O2  =  0.333 
        self.dt  =  0.05  
        #s 
        ### TUNING CONSTANTS 
        self.k  =  1  
        # control gain 
        self.Kp  =  np . array ([[ 1 , 0 ], 
        [ 0 , 1 ]]) 
        self.D  =  np . array ([[ 1 , 0 ], 
        [ 0 , 1 ]]) 
        ### VARIABLES 
        self.q1  =  q1_0 
        self.q2  =  q2_0 
        self.q1dot  =  q1dot_0 
        self.q2dot  =  q2dot_0 
        self.q1dot2  =  q1dot2_0 
        self.q2dot2  =  q2dot2_0 
        self.x1  =  self.L1 * cos ( self.q1 ) +  self.L2 * cos ( self.q1  +  self.q2 ) +  self.O1 
        self.x2  =  self.L1 * sin ( self.q1 ) +  self.L2 * sin ( self.q1  +  self.q2 ) +  self.O2 
        self.x21  =  self.L1 * cos ( self.q1 ) +  self.O1 
        self.x22  =  self.L1 * sin ( self.q1 ) +  self.O2 
        ### REFERNCE INPUT 
        self.q1_ref  =  q1_ref_0 
        self.q2_ref  =  q2_ref_0 
        self.x1_ref  =  self.L1  *  cos  (  self.q1_ref  ) +  self.L2  *  cos  (  self.q1_ref  +  self.q2_ref  ) +  self.O1 
        self.x2_ref  =  self.L1  *  sin  (  self.q1_ref  ) +  self.L2  *  sin  (  self.q1_ref  +  self.q2_ref  ) +  self.O2 
        self.x21_ref  =  self.L1  *  cos  (  self.q1_ref  ) +  self.O1 
        self.x22_ref  =  self.L1  *  sin  (  self.q2_ref  ) +  self.O2 
        self.update_state_matrices  () 
    def  update_state_matrices  (  self  ): 
        self.M  =  np.array  ([ [  self.p1  +  2  *  self.p3  *  cos  (  self.q2  ) ,  self.p2  +  self.p3  *  cos  (  self.q2  ) ] , 
        [  self.p2  +  self.p3  *  cos  (  self.q2  ) , self.p2  ] ] ) 
        self.C  =  np.array  ([ [ -  self.q2dot, self.q1dot  +  self.q2dot  ] , 
        [  self.q1dot, 0  ] ] ) *  self.p3  *  sin  (  self.q2  ) 
        self.g  =  np.array  ([ [  0  ] , 
        [  0  ] ] ) 
        self.Omega  =  np.array  ([ [  self.x1  ] , 
        [  self.x2  ] ] ) 
        self.omega2  =  np.array  ([ [  self.x21  ] , 
        [  self.x22  ] ] ) 
        self.q_ref  =  np.array  ([ [  self.q1_ref  ] , 
        [  self.q2_ref  ] ] ) 
        self.x_ref  =  np.array  ([[  self.x1_ref  ], 
        [  self.x2_ref  ]]) 
        self.q  =  np.array  ([ [  self.q1  ] , 
        [  self.q2  ] ] ) 
        self.qdot  =  np.array  ([ [  self.q1dot  ] , 
        [  self.q2dot  ] ] ) 
        self.qdot2  =  np.array  ([ [  self.q1dot2  ] , 
        [  self.q2dot2  ] ] ) 
        self.E  =  np.copy  (  self.q  -  self.q_ref  ) 
        self.Edot  =  np.copy  (  self.qdot  ) 
        self.Edot2  =  np.copy  (  self.qdot2  ) 
        self.v_input  = -  self.k  *  np.array  ([ [  tanh  (  self.Edot[0][0])  ] , 
        [  tanh  (  self.Edot[1][0])  ] ] ) 
        self.U  =  self.g  -  np.dot  (  self.Kp, self.E  ) +  self.v_input 
    def  advance_and_update_states  (  self  ): 
        self.qdot2  =  np.dot  (  inv  (  self.M  ),-  np.dot  (  self.Kp, self.E  ) -  np.dot  (  self.D, self.Edot  ) 
        -  np.dot  (  self.C, self.Edot  ) +  self.v_input  ) 
        self.qdot  =  self.qdot  +  self.qdot2  *  self.dt 
        self.q  =  self.q  +  self.qdot  *  self.dt 
        self.q1, self.q2  =  self.q[0][0]  , self.q[1][0] 
        self.q1dot, self.q2dot  =  self.qdot[0][0] ,  self.q[1][0] 
        self.q1dot2, self.q2  =  self.qdot2[0][0],  self.q[1][0] 
        self.x1  =  self.L1  *  cos  (  self.q1  ) +  self.L2  *  cos  (  self.q1  +  self.q2  ) +  self.O1 
        self.x2  =  self.L1  *  sin  (  self.q1  ) +  self.L2  *  sin  (  self.q1  +  self.q2  ) +  self.O2 
        self.x21  =  self.L1  *  cos  (  self.q1  ) +  self.O1 
        self.x22  =  self.L1  *  sin  (  self.q1  ) +  self.O2 
        self.update_state_matrices  () 
    def  simulate  (  self, t_f  ): 
        self.t  =  np.linspace  (  0, t_f, int  (  t_f  /  self.dt  )) 
        self.u_array  = [] 
        self.u_1_array  = [] 
        self.u_2_array  = [] 
        self.v_array  = [] 
        self.v_1_array  = [] 
        self.v_2_array  = [] 
        self.q_array  = [] 
        self.q_r_array  = [] 
        self.q_1_array  = [] 
        self.q_1_r_array  = [] 
        self.q_2_array  = [] 
        self.q_2_r_array  = [] 
        self.x_array  = [] 
        self.x_r_array  = [] 
        self.x_1_array  = [] 
        self.x_2_array  = [] 
        self.x_1_r_array  = [] 
        self.x_2_r_array  = [] 
        self.x02_1_array  = [] 
        self.x02_2_array  = [] 
        self.x02_1_r_array  = [] 
        self.x02_2_r_array  = [] 
        for  _  in  range  (  len  (  self.t  )): 
            self.u_array.append  (  np.linalg.norm  (  self.U  )) 
            self.u_1_array.append  (  self.U  [  0  ]) 
            self.u_2_array.append  (  self.U  [  1  ]) 
            self.v_array.append  (  np.linalg.norm  (  self.v_input  )) 
            self.v_1_array.append  (  self.v_input  [  0  ]) 
            self.v_2_array.append  (  self.v_input  [  1  ]) 
            self.q_array.append  (  np.linalg.norm  (  self.q  )) 
            self.q_r_array.append  (  np.linalg.norm  (  self.q_ref  )) 
            self.q_1_array.append  (  self.q  [  0  ]) 
            self.q_2_array.append  (  self.q  [  1  ]) 
            self.q_1_r_array.append  (  self.q1_ref  ) 
            self.q_2_r_array.append  (  self.q2_ref  ) 
            self.x_r_array.append  (  self.x_ref  ) 
            self.x_1_r_array.append  (  self.x1_ref  ) 
            self.x_2_r_array.append  (  self.x2_ref  ) 
            self.x_array.append  (  self.Omega  ) 
            self.x_1_array.append  (  self.x1  ) 
            self.x_2_array.append  (  self.x2  ) 
            self.x02_1_array.append  (  self.x21  ) 
            self.x02_2_array.append  (  self.x22  ) 
            self.x02_1_r_array.append  (  self.x21_ref  ) 
            self.x02_2_r_array.append  (  self.x22_ref  ) 
            self.advance_and_update_states  () 
    def  plot  (  self  ): 
        fig, ax  =  plt.subplots  (  1, 4  ) 
        # ax[0].plot(self.t, self.u_array, '-c') 
        ax[0].plot  (  self.t, self.u_1_array, '-g'  ) 
        ax[0].plot  (  self.t, self.u_2_array, '-r'  ) 
        ax[0].legend  ([  "u1", "u2"  ]) 
        ax[0].set_ylabel  (  "u"  ) 
        ax[0].set_xlabel  (  "t"  ) 
        ax[0].grid  () 
        ax[0].set_title  (  "Control input u v/s iteration t"  ) 
        # ax[1].plot(self.t, self.q_array, '-b') 
        ax[1].plot  (  self.t, self.q_1_array, '-g'  ) 
        ax[1].plot  (  self.t, self.q_2_array, '-r'  ) 
        ax[1].plot  (  self.t, self.q_1_r_array, ':b'  ) 
        ax[1].plot  (  self.t, self.q_2_r_array, ':k'  ) 
        ax[1].legend  ([  "q1", "q2", "q1_ref", "q2_ref"  ]) 
        ax[1].set_ylabel  (  "q"  ) 
        ax[1].set_xlabel  (  "t"  ) 
        ax[1].grid  () 
        ax[1].set_title  (  "Angle q vs iteration t"  ) 
        # ax[2].plot(self.t, self.x_array, '-b') 
        ax[2].plot  (  self.t, self.x_1_array, '-g'  ) 
        ax[2].plot  (  self.t, self.x_2_array, '-r'  ) 
        ax[2].plot  (  self.t, self.x_1_r_array, ':b'  ) 
        ax[2].plot  (  self.t, self.x_2_r_array, ':k'  ) 
        ax[2].legend  ([  "x1", "x2", "x1_ref", "x2_ref"  ]) 
        ax[2].set_ylabel  (  "x"  ) 
        ax[2].set_xlabel  (  "t"  ) 
        ax[2].grid  () 
        ax[2].set_title  (  "Cartesian coordinate x vs iteration t"  ) 
        # ax[3].plot(self.t, self.v_array, '-b') 
        ax[3].plot  (  self.t, self.v_1_array, '-g'  ) 
        ax[3].plot  (  self.t, self.v_2_array, '-r'  ) 
        ax[3].legend  ([  "v1", "v2"  ]) 
        ax[3].set_ylabel  (  "v"  ) 
        ax[3].set_xlabel  (  "t"  ) 
        ax[3].grid  () 
        ax[3].set_title  (  "Passivity control input of v vs iteration t"  ) 
        # plt.show() 
        fig, ax  =  plt.subplots  (  1, 2  ) 
        ax[0].plot  (  self.x_1_array, self.x_2_array, '-g'  ) 
        ax[0].plot  (  self.x_1_r_array, self.x_2_r_array, 'X-b'  ) 
        ax[0].legend  ([  "(x1, x2)", "(x1_ref, x2_ref)"  ]) 
        ax[0].set_ylabel  (  "x2"  ) 
        ax[0].set_xlabel  (  "x1"  ) 
        ax[0].grid  () 
        ax[0].set_title  (  "Cartesian coordinate x2 vs x1"  ) 
        ax[1].plot  (  self.q_1_array, self.q_2_array, '-g'  ) 
        ax[1].plot  (  self.q_1_r_array, self.q_2_r_array, 'X-b'  ) 
        ax[1].legend  ([  "(q1, q2)", "(q1_ref, q2_ref)"  ]) 
        ax[1].set_ylabel  (  "q2"  ) 
        ax[1].set_xlabel  (  "q1"  ) 
        ax[1].grid  () 
        ax[1].set_title  (  "Angle q2 vs q1"  ) 
    # plt.show() 
    def  animate_this  (  self  ): 
        fig  =  plt.figure  () 
        ax  =  fig.add_subplot  (  111, autoscale_on  =  False, xlim  =(-  2, 2  ),  ylim  =(-  2, 2  )) 
        ax.set_aspect  (  'equal'  ) 
        ax.grid  () 
        line  , =  ax.plot  ([], [],  'o-', lw  =  2  ) 
        time_template  =  'time =  %.1f  s' 
        time_text  =  ax.text  (  0.05, 0.9, '', transform  =  ax.transAxes  ) 
        def  init  (): 
            line.set_data  ([], []) 
            time_text  .set_text(  ''  ) 
            return  line, time_text 
        def  animate  (  i  ): 
            thisx  = [  self.O1, self.x02_1_array[i],  self.x_1_array[i]] 
            thisy  = [  self.O2, self.x02_2_array[i],  self.x_2_array[i]] 
            line.set_data  (  thisx, thisy  ) 
            time_text  .set_text(  time_template  % (  i  *  self.dt  )) 
            return  line, time_text 
        ani  =  animation.FuncAnimation  (  fig, animate, np.arange  (  1, len  (  self.x_1_array  )), 
        interval  =  25, blit  =  True, init_func  =  init  ) 
        plt.show  () 
if  __name__  ==  '__main__'  : 
    q_ref  =  np.array  ([[  3  ], [  0  ]]) 
    q_0  =  np.array  ([[  3  ], [  1  ]]) 
    simulation_time  =  50 
    tlm1  =  RoboticManipulator  (  q_ref[0][0], q_ref[1][0], q_0[0][0]  , q_0[1][0] ) 
    tlm1.simulate  (  simulation_time  ) 
    tlm1.plot  () 
    tlm1.animate_this  ()