import numpy as np
from keras import backend as K
from keras.engine.topology import Layer
import math


def generate(output, numComponents=24, outputDim=1, M=1):
    testSize = len(output)
    out_pi = output[:,:numComponents]
    out_sigma = output[:,numComponents:2*numComponents]
    out_mu = output[:,2*numComponents:]
    out_mu = np.reshape(out_mu, [-1, numComponents, outputDim])
    out_mu = np.transpose(out_mu, [1,0,2])
    # use softmax to normalize pi into prob distribution
    max_pi = np.amax(out_pi, 1, keepdims=True)
    out_pi = out_pi - max_pi
    out_pi = np.exp(out_pi)
    normalize_pi = 1 / (np.sum(out_pi, 1, keepdims=True))
    out_pi = normalize_pi * out_pi
    # use exponential to make sure sigma is positive
    out_sigma = np.exp(out_sigma)
    result = np.random.rand(testSize, M, outputDim)
    rn = np.random.randn(testSize, M)
    mu = 0
    std = 0
    idx = 0
    for j in range(0, M):
        for i in range(0, testSize):
            for d in range(0, outputDim):
                idx = np.random.choice(numComponents, 1, p=out_pi[i])
                mu = out_mu[idx,i,d]
                std = out_sigma[i, idx]
                result[i, j, d] = mu + rn[i, j]*std
    return result


def get_mixture_coef(output, numComponents=24, outputDim=1):
    out_pi = output[:,:numComponents]
    out_sigma = output[:,numComponents:2*numComponents]
    out_mu = output[:,2*numComponents:]
    out_mu = K.reshape(out_mu, [-1, numComponents, outputDim])
    out_mu = K.permute_dimensions(out_mu,[1,0,2])
    # use softmax to normalize pi into prob distribution
    max_pi = K.max(out_pi, axis=1, keepdims=True)
    out_pi = out_pi - max_pi
    out_pi = K.exp(out_pi)
    normalize_pi = 1 / K.sum(out_pi, axis=1, keepdims=True)
    out_pi = normalize_pi * out_pi
    # use exponential to make sure sigma is positive
    out_sigma = K.exp(out_sigma)
    return out_pi, out_sigma, out_mu


def tf_normal(y, mu, sigma):
    oneDivSqrtTwoPI = 1 / math.sqrt(2*math.pi)
    result = y - mu
    result = K.permute_dimensions(result, [2,1,0])
    result = result * (1 / (sigma + 1e-8))
    result = -K.square(result)/2
    result = K.exp(result) * (1/(sigma + 1e-8))*oneDivSqrtTwoPI
    result = K.prod(result, axis=[0])
    return result


def get_lossfunc(out_pi, out_sigma, out_mu, y):
    result = tf_normal(y, out_mu, out_sigma)
    result = result * out_pi
    result = K.sum(result, axis=1, keepdims=True)
    result = -K.log(result + 1e-8)
    return K.mean(result)


def mdn_loss(numComponents=24, outputDim=1):
    def loss(y, output):
        out_pi, out_sigma, out_mu = get_mixture_coef(output, numComponents, outputDim)
        return get_lossfunc(out_pi, out_sigma, out_mu, y)
    return loss


class MixtureDensity(Layer):
    def __init__(self, kernelDim, numComponents, hiddenDim=24, **kwargs):
        """

        :param kernelDim: Number of features you're predicting
        :param numComponents: Number of Gaussian distributions
        :param hiddenDim: Size of the hidden layer of the mixture model
        :param kwargs:
        """
        self.hiddenDim = hiddenDim
        self.kernelDim = kernelDim
        self.numComponents = numComponents
        super(MixtureDensity, self).__init__(**kwargs)

    def build(self, inputShape):
        self.inputDim = inputShape[1]
        self.outputDim = self.numComponents * (2+self.kernelDim)
        self.Wh = K.variable(np.random.normal(scale=0.5,size=(self.inputDim, self.hiddenDim)))
        self.bh = K.variable(np.random.normal(scale=0.5,size=(self.hiddenDim)))
        self.Wo = K.variable(np.random.normal(scale=0.5,size=(self.hiddenDim, self.outputDim)))
        self.bo = K.variable(np.random.normal(scale=0.5,size=(self.outputDim)))

        self.trainable_weights = [self.Wh,self.bh,self.Wo,self.bo]

    def call(self, x, mask=None):
        hidden = K.tanh(K.dot(x, self.Wh) + self.bh)
        output = K.dot(hidden,self.Wo) + self.bo
        return output

    def get_output_shape_for(self, inputShape):
        return (inputShape[0], self.outputDim)