import numpy as np
import re
import matplotlib.pyplot as plt
from numpy import cos,sin,arctan2,pi
from numpy.linalg import inv
np.set_printoptions(suppress=True, threshold=np.inf, linewidth=np.inf)
import ipdb

def wraptoPi(angles):
    """Wrap angles  to [-pi,pi]

    Args:
        angles (np.ndarray/float): [Input angles in radians]

    Returns:
        [np.ndarray/float]: [Wrapped around angles]
    """
    xwrap=np.remainder(angles, 2*pi)
    mask = np.abs(xwrap)>pi
    if(type(angles)==np.ndarray):
        xwrap[mask] -= 2*pi * np.sign(xwrap[mask])
    else:
        xwrap-= 2*pi*np.sign(xwrap)*mask
    return xwrap

def drawCovEllipse(c, cov, setting):
    """Draw the Covariance ellipse given the mean and covariance

    :c: Ellipse center
    :cov: Covariance matrix for the state
    :returns: None

    """
    U, s, Vh = np.linalg.svd(cov)
    a, b = s[0], s[1]
    vx, vy = U[0, 0], U[0, 1]
    theta = np.arctan2(vy, vx)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    phi = np.arange(0, 2*np.pi, np.pi/50)
    rot = []
    for i in range(100):
        rect = (np.array([3*np.sqrt(a)*np.cos(phi[i]), 3*np.sqrt(b)*np.sin(phi[i])]))[:, None]
        rot.append(R @ rect + c)

    rot = np.asarray(rot)
    plt.plot(rot[:, 0], rot[:, 1], c=setting, linewidth=0.75)


def drawTrajAndMap(X, last_X, P, t):
    """Draw Trajectory and map

    :X: Current state
    :last_X: Previous state
    :P: Covariance
    :t: timestep
    :returns: None

    """
    plt.ion()
    drawCovEllipse(X[0:2], P[0:2, 0:2], 'b')
    plt.plot([last_X[0], X[0]], [last_X[1], X[1]], c='b', linewidth=0.75)
    plt.plot(X[0], X[1], '*b')

    if t == 0:
        for k in range(6):
            drawCovEllipse(X[3 + k*2:3 + k*2+2], P[3 + k*2:3 + 2*k + 2, 3 + 2*k:3 + 2*k + 2], 'r')
    else:
        for k in range(6):
            drawCovEllipse(X[3 + k*2:3 + k*2+2], P[3 + 2*k:3 + 2*k + 2, 3 + 2*k:3 + 2*k + 2], 'g')

    plt.draw()
    plt.waitforbuttonpress(0)


def drawTrajPre(X, P):
    """ Draw trajectory for Predicted state and Covariance

    :X: Prediction vector
    :P: Prediction Covariance matrix
    :returns: None

    """
    drawCovEllipse(X[0:2], P[0:2, 0:2], 'm')
    plt.draw()
    plt.waitforbuttonpress(0)

def main():
    """Main function for EKF

    :arg1: TODO
    :returns: TODO

    """
    # TEST: Setup uncertainty parameters
    sig_x = 0.025
    sig_y = 0.01
    sig_alpha = 0.01
    sig_beta = 0.01 #0.1
    sig_r = 0.016 #0.08;0.16

    # Generate variance from standard deviation
    sig_x2 = sig_x**2
    sig_y2 = sig_y**2
    sig_alpha2 = sig_alpha**2
    sig_beta2 = sig_beta**2
    sig_r2 = sig_r**2

    # Open data file
    data_file = open("../../data/data.txt", 'r')

    # Read the first measurement data
    line = data_file.readline()
    fields = re.split('[\t ]', line)[:-1]
    arr = np.array([float(field) for field in fields])
    measure = arr[:, None]
    t = 1

    # Setup control and measurement covariance
    control_cov = np.diag([sig_x2, sig_y2, sig_alpha2])
    # measure_cov = np.diag([sig_beta2,sig_r2])
    measure_cov = np.diag([sig_r2,sig_beta2])

    # Setup the initial pose vector and pose uncertainty
    pose = (np.array([0, 0, 0]))[:, None]
    pose_cov = np.diag([0.02**2, 0.02**2, 0.1**2])

    # TODO: Setup the initial landmark estimates landmark[] and covariance matrix landmark_cov[]
    # Hint: use initial pose with uncertainty and first measurement
    k=6
    
    #landmark means
    landmarks=np.zeros_like(measure)     #12x1
    ranges,bearings = measure[1::2,:],measure[::2,:] #6x1,6x1
    landmarks[::2,:] = pose[0] + ranges*cos(wraptoPi(bearings+pose[2]))
    landmarks[1::2,:] = pose[0] + ranges*sin(wraptoPi(bearings+pose[2]))
    landmark_cov = np.zeros((2*k,2*k))
    for idx in range(k):
        beta = bearings[idx,0]
        initH = np.array([  [1, 0, -ranges[idx,0] * sin(beta)],
                            [ 0, 1, ranges[idx,0] * cos(beta)]],dtype=object)
        initQ = np.array([[-ranges[idx,0] * sin(beta), cos(beta)], 
                [ranges[idx,0] * cos(beta), sin(beta)]],dtype=object)
        landmark_cov[idx * 2  : idx * 2 + 2, idx * 2  : idx * 2 + 2] = initH @ pose_cov @ initH.T + initQ @ measure_cov @ initQ.T
    # landmark_cov = np.diag([0.02]*2*k) #6x2 landmark poses with inital covariance set as 0.02


    ##############################################################
    ################## Write your code here ######################
    ##############################################################

    # Setup state vector x with pose and landmark vector
    X = np.vstack((pose, landmarks))
    # Setup covariance matrix P with pose and landmark covariance
    P = np.block([[pose_cov,           np.zeros((3, 2*k))],
                  [np.zeros((2*k, 3)),       landmark_cov]])

    # Plot initial state and covariance
    last_X = X.copy()
    drawTrajAndMap(X, last_X, P, 0)

    # Read file sequentially for controls
    # and measurements
    
    G=np.eye(3)
    G_reshaped = np.eye(P.shape[0])
    control_cov_reshaped= np.zeros_like(P)
    control_cov_reshaped[:3,:3] = control_cov
    measure_cov_reshaped= np.zeros_like(P)
    measure_cov_reshaped[:2,:2] = measure_cov
    

    for log_idx,line in enumerate(data_file):
        print("Log number: {}".format(log_idx+1))
        fields = re.split('[\t ]', line)[:-1]
        arr = np.array([float(field) for field in fields])
        if arr.shape[0] == 2:
            d, alpha = arr[0], arr[1]
            control = (np.array([d, alpha]))[:, None]

            # TODO: Predict step
            # (Notice: predict state x_pre[] and covariance P_pre[] using input control data and control_cov[])
            
            
            ##############################################################
            ################## Write your code here ######################
            ##############################################################
            #mean at prediction step
            last_theta= last_X[2]
            pose_mean_pred = last_X[:3] + np.array([d*cos(last_theta),d*sin(last_theta),alpha],dtype=object).reshape(3,1)
            pose_mean_pred[2]=wraptoPi(pose_mean_pred[2])
            #Pose Jacobian
            G[:2,2]= -d*sin(last_theta),d*cos(last_theta)
            #covariance at prediction step
            
            G_reshaped[:3,:3] = G
            
            P_pre =  G_reshaped @ P @ G_reshaped.T + control_cov_reshaped
            
            X_pre= last_X.copy()
            X_pre[:3,:]=pose_mean_pred
            

            # Draw predicted state X_pre and Covariance P_pre
            drawTrajPre(X_pre, P_pre)

        # Read the measurement data
        else:
            measure = (arr)[:, None]

            # TODO: Correction step
            # (Notice: Update state X[] and covariance P[] using the input measurement data and measurement_cov[])

            ##############################################################
            ################## Write your code here ######################
            ##############################################################
            x,y,theta = X_pre[:3]
            
            for idx in range(0,measure.shape[0],2):
                
                delx,dely = ( X_pre[idx+3:idx+5,:] - X_pre[:2,:]).squeeze()
                q = (delx)**2 + (dely)**2
                _range=np.sqrt(q)
                bearing = arctan2(dely,delx)-theta
                bearing = wraptoPi(np.array(bearing))[0]
                z_hat = np.array([_range,wraptoPi(bearing)]).reshape(2,1)
                z=np.flip(measure[idx:idx+2])
                ## Hp and Hl concatenated 2x5 matrix
                
                
                Hp = (1/q) * np.array([    [-_range*delx , -_range*dely, 0, _range*delx , _range*dely],
                                              [dely,    -delx ,      -q ,   -dely,        delx]])
                H=np.zeros((2,X.shape[0]))
                H[:2,:3]=Hp[:2,:3]
                H[:2,idx+3:idx+5]=Hp[:2,3:5]

                K = P_pre@ H.T @   inv(H@P_pre@H.T + measure_cov)
                X_pre += K@(z-z_hat)
                P_pre = (np.eye(P.shape[0]) - K@H)@P_pre
            
            

            
            X = X_pre.copy()
            
            P = P_pre.copy()
            drawTrajAndMap(X, last_X, P, t)
            last_X = X.copy()
            t += 1
    # plt.title('Final Trajectory')
    landmark_gt = np.array([3, 6, 3, 12, 7 ,8 ,7, 14, 11, 6 ,11 ,12])
    plt.scatter(landmark_gt[::2], landmark_gt[1::2], marker='*')
    drawTrajAndMap(X, last_X, P, t)
    # EVAL: Plot ground truth landmarks
    plt.savefig('final_trajectory_xyarb_{}_{}_{}_{}_{}.png'.format(sig_x,sig_y,sig_alpha,sig_r,sig_beta))
    ##############################################################
    ################## Write your code here ######################
    ##############################################################
    
    plt.figure()
    plt.title("Final Landmarks_xyarb_{}_{}_{}_{}_{}.png".format(sig_x,sig_y,sig_alpha,sig_r,sig_beta))
    
    plt.scatter(landmark_gt[::2], landmark_gt[1::2], marker='*')
    for i in range(k):
        drawCovEllipse(X[3 + i*2:3 + i*2+2], P[3 + i*2:3 + 2*i + 2, 3 + 2*i:3 + 2*i + 2], 'r')
        
    
    plt.savefig('finallandmarks.png')
    
    ####Evaluation#####
    est_landmarks= X[3:,0]
    dx,dy = est_landmarks[::2] - landmark_gt[::2] , est_landmarks[1::2] - landmark_gt[1::2]
    #Euclidean-
    euc_distances = np.sqrt(dx**2 + dy**2)
    print("Final Euclidean distances :\n{}".format(euc_distances))
    #Mahalanobis-
    diff = np.abs(np.vstack((dx,dy))) #2x6
    maha_distances=[]
    for idx in range(k):
        dist= diff[:,idx].reshape(2,1)
        sigma = P[2*idx+3:2*idx+2+3,2*idx+3:2*idx+2+3]
        distance = np.sqrt((dist.T@ inv(sigma)@ dist)[0,0])
        maha_distances.append(distance)
    print("Final Mahalanobis distances :\n{}".format(maha_distances))
    # ipdb.set_trace()




if __name__ == "__main__":
    ipdb.set_trace()
    main()
