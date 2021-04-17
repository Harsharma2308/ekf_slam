from numpy import cos,sin,arctan2,pi
import numpy as np
import matplotlib.pyplot as plt


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