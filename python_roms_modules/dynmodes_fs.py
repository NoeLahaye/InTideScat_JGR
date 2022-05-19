# -*- coding: utf-8 -*-
"""Ocean vertical dynamic modes exercise code for Mathematical Modelling of
Geophysical Fluids MPE2013 Workshop at African Institute for Mathematical Sciences.
Adopted from https://bitbucket.org/douglatornell/aims-workshop
by Doug Latornell.
"""
import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg as la

grav = 9.81     # multiply Nsq by a constant to virtualy change these values
                # (and divide eigenvalues by Nsq, i.e. multiply ce by N)

def dynmodes(Nsq, depth, nmodes):
    """Calculate the 1st nmodes ocean dynamic vertical modes
    given a profile of Brunt-Vaisala (buoyancy) frequencies squared.

    Based on
    http://woodshole.er.usgs.gov/operations/sea-mat/klinck-html/dynmodes.html
    by John Klinck, 1999.

    :arg Nsq: Brunt-Vaisala (buoyancy) frequencies squared in [1/s^2]
    :type Nsq: :class:`numpy.ndarray`

    :arg depth: Depths in [m]
    :type depth: :class:`numpy.ndarray`

    :arg nmodes: Number of modes to calculate
    :type nmodes: int

    :returns: :obj:`(wmodes, pmodes, rmodes), ce, (dz_w, dz_p)` (vertical velocity modes,
              horizontal velocity modes, vertical density modes), modal speeds,
              (box size of vertical velocity grid, box size of pressure grid)
    :rtype: tuple of :class:`numpy.ndarray`
    """
    if np.all(depth >= 0.):
        z = -depth
    else:
        z = depth

    nmodes = min((nmodes, len(z)-2))
    # 2nd derivative matrix plus boundary conditions
    d2dz2_w, dz_w = build_d2dz2_matrix_w(z)
    # N-squared diagonal matrix
    Nsq_mat = np.diag(Nsq)
    # Solve generalized eigenvalue problem for eigenvalues and vertical
    # velocity modes
    eigenvalues_w, wmodes = la.eigs(d2dz2_w, k=nmodes, M=Nsq_mat, which='SM')
    eigenvalues_w, wmodes = clean_up_modes(eigenvalues_w, wmodes, nmodes)
    # Horizontal velocity modes
    d2dz2_p, dz_p = build_d2dz2_matrix_p(z, Nsq)
    eigenvalues_p, pmodes = la.eigs(d2dz2_p, k=nmodes, which='SM')
    eigenvalues_p, pmodes = clean_up_modes(eigenvalues_p, pmodes, nmodes)
    nmodes = min(pmodes.shape[1], wmodes.shape[1])
    eigenvalues_p, eigenvalues_w, pmodes, wmodes = (
        eigenvalues_p[:nmodes], eigenvalues_w[:nmodes], pmodes[:, :nmodes], wmodes[:, :nmodes])
    
    # Vertical density modes
    rmodes = wmodes * Nsq[:, np.newaxis]
    # Modal speeds
    ce = 1 / np.sqrt(eigenvalues_p)
    print("Mode speeds do correspond: %s" % np.allclose(ce * np.sqrt(eigenvalues_w), 1.))
    # unify sign, that pressure modes are alwas positive at the surface
    modes = unify_sign(wmodes, pmodes, rmodes)
    return modes, ce, (dz_w, dz_p)

def dynmodes_fs(Nsq, z, nmodes, fs=True):
    """Calculate the 1st nmodes ocean dynamic vertical modes, with free-surface top BC,
    given a profile of Brunt-Vaisala (buoyancy) frequencies squared.

    Based on
    http://woodshole.er.usgs.gov/operations/sea-mat/klinck-html/dynmodes.html
    by John Klinck, 1999.

    :arg Nsq: Brunt-Vaisala (buoyancy) frequencies squared in [1/s^2]
    :type Nsq: :class:`numpy.ndarray`

    :arg z: Depths in [m]
    :type z: :class:`numpy.ndarray`

    :arg nmodes: Number of modes to calculate
    :type nmodes: int

    :arg fs: free surface (default) or not (rigid lid)
    :type fs: :class:`bool`

    :returns: :obj:`(wmodes, pmodes), ce, (dz_w, dz_p)` (vertical velocity modes,
              horizontal velocity modes), modal speeds,
              (box size of vertical velocity grid, box size of pressure grid)
    :rtype: tuple of :class:`numpy.ndarray`
    """
    godown = z[0]<0
    if godown: 
        z = z[::-1]     # algebraic values: go from top to bottom
        Nsq = Nsq[::-1]
    else:
        z = -z          # absolute depth: switch to algebraic
    nmodes = min((nmodes, len(z)-2))
    # 2nd derivative matrix plus boundary conditions
    d2dz2_w, dz_w = build_d2dz2_matrix_neww(z,fs)
    # N-squared diagonal matrix -- with modification (M in generalized eigen-problem)
    if len(Nsq) == len(z)-2:
        if fs:
            Nsq_mat = np.diag(r_[1.,Nsq[1:]])
        else:
            Nsq_mat = np.diag(np.r_[Nsq[0],Nsq])
    elif len(Nsq) == len(z):
        if fs:
            Nsq_mat = np.diag(np.r_[1.,Nsq[1:-1]])
        else:
            Nsq_mat = np.diag(Nsq[1:-1])
    else:
        raise ValueError('size of Nsq does not match')
    # Solve generalized eigenvalue problem for eigenvalues and vertical
    # velocity modes
    eigenvalues_w, wmodes = la.eigs(d2dz2_w, k=nmodes, M=Nsq_mat, which='SM')
    eigenvalues_w, wmodes = clean_up_modes(eigenvalues_w, wmodes, nmodes)
    if fs:
        wmodes = np.vstack((wmodes,np.zeros(wmodes.shape[-1])))
    else:
        wmodes = np.vstack((np.zeros(wmodes.shape[-1]),wmodes,np.zeros(wmodes.shape[-1])))
    # Horizontal velocity modes
    d2dz2_p, dz_p = build_d2dz2_matrix_newp(z, Nsq, fs)
    eigenvalues_p, pmodes = la.eigs(d2dz2_p, k=nmodes, which='SM')
    eigenvalues_p, pmodes = clean_up_modes(eigenvalues_p, pmodes, nmodes)
    nmodes = min(pmodes.shape[1], wmodes.shape[1])
    eigenvalues_p, eigenvalues_w, pmodes, wmodes = (
        eigenvalues_p[:nmodes], eigenvalues_w[:nmodes], pmodes[:, :nmodes], wmodes[:, :nmodes])
    
    # Modal speeds
    ce = 1 / np.sqrt(eigenvalues_p)
    print("Mode speeds do correspond: %s" % np.allclose(ce * np.sqrt(eigenvalues_w), 1.))
    # unify sign, that pressure modes are alwas positive at the surface
    modes = unify_sign2(wmodes, pmodes, godown)
    if godown:
        dz_w, dz_p = dz_w[::-1], dz_p[::-1]
    return modes, ce, (-dz_w, -dz_p)

def dynmodes_w(Nsq, z, nmodes, fs=True):
    """Calculate the 1st nmodes ocean dynamic vertical modes, with free-surface top BC,
    given a profile of Brunt-Vaisala (buoyancy) frequencies squared.
    Returns vertical velocity modes normalized such that max(|w|)=1

    Based on
    http://woodshole.er.usgs.gov/operations/sea-mat/klinck-html/dynmodes.html
    by John Klinck, 1999.

    :arg Nsq: Brunt-Vaisala (buoyancy) frequencies squared in [1/s^2]
    :type Nsq: :class:`numpy.ndarray`

    :arg z: Depths in [m], algebraic (<0, first one is bottom) or absolute (first one is surface)
    :type z: :class:`numpy.ndarray`

    :arg nmodes: Number of modes to calculate
    :type nmodes: int

    :arg fs: free-surface (default) or not
    :type fs: :class:`bool`

    :returns: :obj:`wmodes, ce, dz_w` (vertical velocity modes, modal speeds,
              box size of vertical velocity grid)
    :rtype: tuple of :class:`numpy.ndarray`
    """
    godown = z[0]<0
    if godown: 
        z = z[::-1]     # algebraic values: go from top to bottom
        Nsq = Nsq[::-1]
    else:
        z = -z          # absolute depth: switch to algebraic
    nmodes = min((nmodes, len(z)-2))
    # 2nd derivative matrix plus boundary conditions
    d2dz2_w, dz_w = build_d2dz2_matrix_neww(z,fs)
    # N-squared diagonal matrix (M in generalized eigen-problem)
    if fs:
        Nsq_mat = np.diag(np.r_[1.,Nsq[1:-1]])
    else:
        Nsq_mat = np.diag(Nsq[1:-1])
    # Solve generalized eigenvalue problem for eigenvalues and vertical
    # velocity modes
    #eigenvalues, wmodes = la.eigs(d2dz2_w, k=nmodes, M=Nsq_mat, which='SM')
    eigenvalues, wmodes = scipy.linalg.eig(d2dz2_w.toarray(), Nsq_mat)
    eigenvalues, wmodes = clean_up_modes(eigenvalues, wmodes, nmodes)
    if fs:
        wmodes = np.vstack((wmodes*np.sign(wmodes[-1,:])/np.abs(wmodes).max(axis=0)[None,:]\
                    ,np.zeros(wmodes.shape[-1])))
    else:
        wmodes = np.vstack((np.zeros(wmodes.shape[-1])\
                    ,wmodes*np.sign(wmodes[-1,:])/np.abs(wmodes).max(axis=0)[None,:]\
                    ,np.zeros(wmodes.shape[-1])))
    if godown: 
        wmodes = wmodes[::-1,:]
        dz_w = dz_w[::-1]
    return wmodes, 1./np.sqrt(eigenvalues), -dz_w


def unify_sign(wmodes, pmodes, rmodes):
    sig_p = np.sign(pmodes[0, :])
    sig_p[sig_p == 0.] = 1.
    sig_w = np.sign(wmodes[0, :] - wmodes[1, :])
    sig_w[sig_w == 0.] = 1.
    pmodes = sig_p * pmodes
    wmodes = sig_w * wmodes
    rmodes = sig_w * rmodes
    return wmodes, pmodes, rmodes

def unify_sign2(wmodes, pmodes, godown=False):
    sig_p = np.sign(pmodes[0, :])
    sig_p[sig_p == 0.] = 1.
    sig_w = np.sign(wmodes[0, :] - wmodes[1, :])
    sig_w[sig_w == 0.] = 1.
    if godown:
        pmodes = sig_p * pmodes[::-1,:]
        wmodes = sig_w * wmodes[::-1,:]
    else:
        pmodes = sig_p * pmodes
        wmodes = sig_w * wmodes
    return wmodes, pmodes 


def build_d2dz2_matrix_w(z):
    """Build the matrix that discretizes the 2nd derivative
    over the vertical coordinate, and applies the boundary conditions for
    w-mode.

    :arg z: Depths in [m], positive up
    :type depth: :class:`numpy.ndarray`

    :returns: :obj:`(d2dz2, nz, dz)` 2nd derivative matrix
              (:class:`numpy.ndarray`),
              number of vertical coordinate grid steps,
              and array of vertical coordinate grid point spacings
              (:class:`numpy.ndarray`)
    :rtype: tuple
    """
    # Size (in [m]) of vertical coordinate grid steps
    dz = np.diff(z)
    z_mid = z[1:] - dz / 2.
    dz_mid = np.diff(z_mid)

    d0 = np.r_[-1., (1. / dz[:-1] + 1. / dz[1:]) / dz_mid, -1.]
    d1 = np.r_[0., 0., -1. / dz[1:] / dz_mid]
    dm1 = np.r_[-1. / dz[:-1] / dz_mid, 0., 0.]
    diags = np.vstack((d0, d1, dm1))

    d2dz2 = scipy.sparse.dia_matrix((diags, (0, 1, -1)), shape=(len(z), len(z)))
#    pcolor(d2dz2.toarray()[:10, :10]); colorbar(); title('d2dz2')
    dz_mid = np.r_[dz[0] / 2., dz_mid, dz[-1] / 2.]
    return d2dz2, dz_mid

def build_d2dz2_matrix_neww(z, fs=False):
    """Build the matrix that discretizes the 2nd derivative
    over the vertical coordinate, and applies the boundary conditions for
    w-mode.

    :arg z: Depths in [m], positive up
    :type depth: :class:`numpy.ndarray`

    :returns: :obj:`(d2dz2, nz, dz)` 2nd derivative matrix
              (:class:`numpy.ndarray`),
              number of vertical coordinate grid steps,
              and array of vertical coordinate grid point spacings
              (:class:`numpy.ndarray`)
    :rtype: tuple
    """
    # Size (in [m]) of vertical coordinate grid steps
    dz = np.diff(z)
    z_mid = z[1:] - dz / 2.
    dz_mid = np.diff(z_mid)

    if fs:
        d0 = np.r_[-1./dz[0]/grav, (1. / dz[:-1] + 1. / dz[1:]) / dz_mid]
        d1 = np.r_[0., +1./dz[0]/grav, -1. / dz[1:-1] / dz_mid[:-1]]
        dm1 = np.r_[-1. / dz[:-1] / dz_mid, 0.]
    else:
        d0 = (1. / dz[:-1] + 1. / dz[1:]) / dz_mid
        d1 = np.r_[0., -1. / dz[1:-1] / dz_mid[:-1]]
        dm1 = np.r_[-1. / dz[1:-1] / dz_mid[1:], 0]
    diags = np.vstack((d0, d1, dm1))
    nz = len(z) - 2 + int(fs)

    d2dz2 = scipy.sparse.dia_matrix((diags, (0, 1, -1)), shape=(nz, nz))
#    pcolor(d2dz2.toarray()[:10, :10]); colorbar(); title('d2dz2')
    dz_mid = np.r_[dz[0] / 2., dz_mid, dz[-1] / 2.]
    return d2dz2, dz_mid

def build_d2dz2_matrix_p(z, Nsq):
    """Build the matrix that discretizes the 2nd derivative
    over the vertical coordinate, and applies the boundary conditions for 
    p-mode.

    :arg z: Depths in [m], positive up
    :type depth: :class:`numpy.ndarray`

    :returns: :obj:`(d2dz2, nz, dz)` 2nd derivative matrix
              (:class:`numpy.ndarray`),
              number of vertical coordinate grid steps,
              and array of vertical coordinate grid point spacings
              (:class:`numpy.ndarray`)
    :rtype: tuple
    """
    dz = np.diff(z)
    z_mid = z[1:] - dz / 2.
    dz_mid = np.diff(z_mid)
    dz_mid = np.r_[dz[0] / 2., dz_mid, dz[-1] / 2.]
    
    Ndz = Nsq * dz_mid

    d0 = np.r_[1. / Ndz[1] / dz[0],
               (1. / Ndz[2:-1] + 1. / Ndz[1:-2]) / dz[1:-1],
               1. / Ndz[-2] / dz[-1]]
    d1 = np.r_[0., -1. / Ndz[1:-1] / dz[:-1]]
    dm1 = np.r_[-1. / Ndz[1:-1] / dz[1:], 0.]

    diags = np.vstack((d0, d1, dm1))
    d2dz2 = scipy.sparse.dia_matrix((diags, (0, 1, -1)), shape=(len(z)-1, len(z)-1))
    return d2dz2, dz

def build_d2dz2_matrix_newp(z, Nsq, fs=False):
    """Build the matrix that discretizes the 2nd derivative
    over the vertical coordinate, and applies the boundary conditions for 
    p-mode.

    :arg z: Depths in [m], positive up
    :type depth: :class:`numpy.ndarray`

    :returns: :obj:`(d2dz2, nz, dz)` 2nd derivative matrix
              (:class:`numpy.ndarray`),
              number of vertical coordinate grid steps,
              and array of vertical coordinate grid point spacings
              (:class:`numpy.ndarray`)
    :rtype: tuple
    """
    dz = np.diff(z)
    z_mid = z[1:] - dz / 2.
    dz_mid = np.diff(z_mid)
    dz_mid = np.r_[dz[0] / 2., dz_mid, dz[-1] / 2.]
    
    Ndz = Nsq * dz_mid

    d0 = np.r_[1. / Ndz[1] / dz[0],
               (1. / Ndz[2:-1] + 1. / Ndz[1:-2]) / dz[1:-1],
               1. / Ndz[-2] / dz[-1]]
    d1 = np.r_[0., -1. / Ndz[1:-1] / dz[:-1]]
    dm1 = np.r_[-1. / Ndz[1:-1] / dz[1:], 0.]
    if fs: 
        d0[0] += -1/grav/dz[0]

    diags = np.vstack((d0, d1, dm1))
    d2dz2 = scipy.sparse.dia_matrix((diags, (0, 1, -1)), shape=(len(z)-1, len(z)-1))
    return d2dz2, dz


def clean_up_modes(eigenvalues, wmodes, nmodes):
    """Exclude complex-valued and near-zero/negative eigenvalues and their
    modes. Sort the eigenvalues and mode by increasing eigenvalue magnitude,
    truncate the results to the number of modes that were requested,
    and convert the modes from complex to real numbers.

    :arg eigenvalues: Eigenvalues
    :type eigenvalues: :class:`numpy.ndarray`

    :arg wmodes: Modes
    :type wmodes: :class:`numpy.ndarray`

    :arg nmodes: Number of modes requested
    :type nmodes: int

    :returns: :obj:`(eigenvalues, wmodes)`
    :rtype: tuple of :class:`numpy.ndarray`
    """
    # Filter out complex-values and small/negative eigenvalues
    # and corresponding modes
    mask = np.logical_and(eigenvalues >= 1e-10, eigenvalues.imag == 0)
    eigenvalues = eigenvalues[mask]
    wmodes = wmodes[:, mask]

    # Sort eigenvalues and modes and truncate to number of modes requests
    index = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[index[:nmodes]]
    wmodes = wmodes[:, index[:nmodes]]
    return eigenvalues.real, wmodes.real


def plot_modes(Nsq, depth, nmodes, wmodes, pmodes, rmodes):
    """Plot Brunt-Vaisala (buoyancy) frequency profile and 3 sets of modes
    (vertical velocity, horizontal velocity, and vertical density) in 4 panes.

    :arg Nsq: Brunt-Vaisala (buoyancy) frequencies squared in [1/s^2]
    :type Nsq: :class:`numpy.ndarray`

    :arg depth: Depths in [m]
    :type depth: :class:`numpy.ndarray`

    :arg wmodes: Vertical velocity modes
    :type wmodes: :class:`numpy.ndarray`

    :arg pmodes: Horizontal velocity modes
    :type pmodes: :class:`numpy.ndarray`

    :arg rmodes: Vertical density modes
    :type rmodes: :class:`numpy.ndarray`

    :arg nmodes: Number of modes to calculate
    :type nmodes: int
    """
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(2, 2, 1)
    # Nsq
    ax.plot(Nsq, -depth)
    ax.ticklabel_format(style='sci', scilimits=(2, 2), axis='x')
    ax.set_ylabel('z')
    ax.set_xlabel('N^2')
    # modes
    mode_sets = [
        # (values, subplot number, x-axis title)
        (wmodes, 2, 'wmodes'),
        (pmodes, 3, 'pmodes'),
        (rmodes, 4, 'rmodes'),
    ]
    for mode_set in mode_sets:
        modes, subplot, title = mode_set
        ax = fig.add_subplot(2, 2, subplot)
        for i in xrange(nmodes):
            ax.plot(modes[i], -depth, label='mode {}'.format(i + 1))
        ax.ticklabel_format(style='sci', scilimits=(3, 3), axis='x')
        ax.set_ylabel('z')
        ax.set_xlabel(title)
        ax.legend(loc='best')


def read_density_profile(filename):
    """Return depth and density arrays read from filename.

    :arg filename: Name of density profile file.
    :type filename: string

    :returns: :obj:`(depth, density)` depths, densities
    :rtype: tuple of :class:`numpy.ndarray`
    """
    depth = []
    density = []
    with open(filename) as f:
        for line in interesting_lines(f):
            deep, rho = map(float, line.split())
            depth.append(deep)
            density.append(rho)
    return np.array(depth), np.array(density)


def interesting_lines(f):
    for line in f:
        if line and not line.startswith('#'):
            yield line


def density2Nsq(depth, density, rho0=1028):
    """Return the Brunt-Vaisala (buoyancy) frequency (Nsq) profile
    corresponding to the given density profile.
    The surface Nsq value is set to the value of the 1st calculated value
    below the surface.
    Also return the depths for which the Brunt-Vaisala (buoyancy) frequencies squared
    were calculated.

    :arg depth: Depths in [m]
    :type depth: :class:`numpy.ndarray`

    :arg density: Densities in [kg/m^3]
    :type density: :class:`numpy.ndarray`

    :arg rho0: Reference density in [kg/m^3]; defaults to 1028
    :type rho0: number

    :returns: :obj:`(Nsq_depth, Nsq)` depths for which the Brunt-Vaisala
              (buoyancy) frequencies squared were calculated,
              Brunt-Vaisala (buoyancy) frequencies squared
    :rtype: tuple of :class:`numpy.ndarray`
    """
    grav_acc = 9.8  # m / s^2
    Nsq = np.zeros_like(density)
    Nsq[1:] = np.diff(density) * grav_acc / (np.diff(depth) * rho0)
    Nsq[0] = Nsq[1]
    Nsq[Nsq < 0] = 0
    Nsq_depth = np.zeros_like(depth)
    Nsq_depth[1:] = (depth[:depth.size - 1] + depth[1:]) / 2
    return Nsq_depth, Nsq

