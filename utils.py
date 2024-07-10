import torch
import torch_dct as dct

from torchvision.transforms import functional as F

def flip_odd_lines(matrix):
    """
    Flip odd lines of a matrix
    """
    matrix = matrix.clone()

    matrix[..., 1::2, :] = matrix[..., 1::2, :].flip(-1)

    return matrix

rotate = lambda m, r: torch.rot90(m, r, [-2, -1])
sequency_vec = lambda m: flip_odd_lines(rotate(m, 0)).flatten(start_dim= m.dim()-2)
sequency_mat = lambda v, s: rotate(flip_odd_lines(v.unflatten(-1, s)), 0)


def modulo(x, L):
    positive = x > 0
    x = x % L
    x = torch.where( ( x == 0) &  positive, L, x)
    return x


def center_modulo(x, L):
    return modulo(x + L/2, L) - L/2


def unmodulo(psi):

    psi = torch.nn.functional.pad(psi, (1,1), mode='constant', value=0)
    psi = torch.diff(psi, 1)
    psi = dct.dct(psi, norm='ortho')
    N = psi.shape[-1]
    k = torch.arange(0, N)
    denom = 2*( torch.cos(  torch.pi * k / N  )  -  1  )
    denom[0] = 1.0
    denom = denom.unsqueeze(0).unsqueeze(0) + 1e-7
    psi     = psi / denom   
    psi[..., 0] = 0.0
    psi = dct.idct(psi, norm='ortho')
    return psi

RD = lambda x, L: torch.round( x / L)  * L


def hard_thresholding(x, t):
    return x * (torch.abs(x) > t)


def stripe_estimation(psi, t=0.15):

    dx = torch.diff(psi, 1, dim=-1)
    dy = torch.diff(psi, 1, dim=-2)

    dx = hard_thresholding(dx, t) 
    dy = hard_thresholding(dy, t) 

    dx = F.pad(dx, (1, 0, 1, 0))
    dy = F.pad(dy, (0, 1, 0, 1))


    rho = torch.diff(dx, 1, dim=-1) + torch.diff(dy, 1, dim=-2)
    dct_rho = dct.dct_2d(rho, norm='ortho')


    MX = rho.shape[-2]
    NX = rho.shape[-1]

    I, J = torch.meshgrid(torch.arange(0, MX), torch.arange(0, NX), indexing="ij")
    I = I.to(rho.device)
    J = J.to(rho.device)
    denom = 2 * (torch.cos(torch.pi * I / MX ) + torch.cos(torch.pi * J / NX ) - 2)
    denom = denom.unsqueeze(0).unsqueeze(0)
    denom = denom.to(rho.device)
    dct_phi = dct_rho / denom
    dct_phi[..., 0, 0] = 0
    phi = dct.idct_2d(dct_phi, norm='ortho')
    phi = phi - torch.min(phi)
    # phi = phi - torch.amin(phi, dim=(-1, -2), keepdim=True)
    # phi = RD(phi, 1.0)
    return phi


def recons(m_t, DO=1, L=1.0, vertical=False, t=0.3):

    if vertical:
        m_t = m_t.permute(0, 1, 3, 2)

    shape = m_t.shape[-2:]

    modulo_vec = sequency_vec(m_t)
    res = center_modulo( torch.diff(modulo_vec, n=DO), L) - torch.diff(modulo_vec, n=DO)
    bl = res

    for i in range(DO):
        bl = unmodulo(bl)
        bl = RD(bl, L)

    x_est = bl
    
    x_est = sequency_mat(x_est, shape)   
    x_est = x_est + m_t

    if vertical:
        x_est = x_est.permute(0, 1, 3, 2)

    stripes = stripe_estimation(x_est, t=t)    
    x_est = x_est - stripes

    # if vertical:
    #     x_est = x_est.permute(0, 1, 3, 2)
    x_est = x_est - x_est.min()
    x_est = x_est / x_est.max()
    return x_est 