# -*- coding: utf-8 -*-


"""
    Ex-Post Density Estimation (XPDE).
"""
import torch
import numpy as np


class LatentSpaceSampler(object):
    def __init__(self, encoder):
        self.encoder = encoder
        self.z_cov = None

    def get_z_covariance(self, batches_of_xs):
        """Takes one or more batches of xs of shape batches X data_dims"""

        zs = self.encoder(batches_of_xs).detach().cpu().numpy()

        z_original_shape = zs.shape
        zs = np.reshape(zs, (z_original_shape[0], -1))
        """
            >>> a
            tensor([[ 0.2408,  1.1318, -0.4333],
                    [ 0.0169, -0.9865,  0.0957]])

            >>> b = np.cov(a)
            array([[ 0.61631135, -0.44011707],
                   [-0.44011707,  0.36400009]])
                   
            >>> a.shape
            torch.Size([2, 3])
            
            >>> mean1 = (0.2408 + 1.1318 -0.4333) /3
            0.3131
            
            >>> cov(1, 1) = \sum_i^{n} (X1 - mean1) (X1 - mean1)
            
            >>> (0.2408 - 0.3131) ** 2 + (1.1318 - 0.3131) ** 2 + (-0.4333 - 0.3131) ** 2
            1.23261
            
            >>> n = 2
            
            >>> 1.23261 / (n-1)
            0.616305 
        """
        self.z_cov = np.cov(zs.T)  # https://blog.csdn.net/jeffery0207/article/details/83032325
        # shape of self.z_cov: [bottleneck_factor=16, bottleneck_factor=16]

        return self.z_cov, z_original_shape

    def get_zs(self, batches_of_xs):
        """batches_of_xs are only used to compute variance of Z on the fly"""
        num_smpls = batches_of_xs.shape[0]
        self.z_cov, z_dim = self.get_z_covariance(batches_of_xs)

        try:
            zs_flattened = np.random.multivariate_normal(np.zeros(np.prod(z_dim[1:]), ), cov=self.z_cov,
                                                         size=num_smpls)  # https://www.zhihu.com/question/288946037/answer/649328934
        except np.linalg.LinAlgError as e:
            print(self.z_cov)
            print(e)
            zs_flattened = np.random.multivariate_normal(np.zeros(np.prod(z_dim[1:]), ),
                                                         cov=self.z_cov + 1e-5 * np.eye(self.z_cov.shape[0]),
                                                         size=num_smpls)

        return np.reshape(zs_flattened, (num_smpls,) + z_dim[1:])
