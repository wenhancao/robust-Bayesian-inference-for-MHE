import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import t

# Set global font and size
plt.rc('font', family='serif', size=20)

# Define parameters for the true and outlier distributions
mu_true, sigma_true = 0, 1
mu_outlier, sigma_outlier = 6, 1
N_true = 800
N_outlier = 200

# # Generate data
# np.random.seed(0)
# data_true = np.random.normal(mu_true, sigma_true, N_true)
# data_outlier = np.random.normal(mu_outlier, sigma_outlier, N_outlier)
# data = np.concatenate((data_true, data_outlier))

df = 10  # Adjust this value as needed

# Generate data
np.random.seed(0)
data_true = np.random.normal(mu_true, sigma_true, N_true)
data_outlier = t.rvs(df, loc=mu_outlier, scale=sigma_outlier, size=N_outlier)
data = np.concatenate((data_true, data_outlier))


# Define KL divergence and Beta divergence
def kl_divergence(params):
    mu, sigma = params
    return -np.sum(norm.logpdf(data, mu, sigma))


def beta_divergence(params, beta):
    mu, sigma = params
    if beta == 0:
        return kl_divergence(params)
    else:
        return -(beta + 1) / beta * (np.exp(np.sum(norm.logpdf(data, mu, sigma)) * beta) - 1)


# Perform MLE using KL divergence and Beta divergence
res_kl = minimize(kl_divergence, [0, 1], tol=1e-10, method='Nelder-Mead', bounds=[(-100, 100), (0, 100)])

betas = [0.1]  # Decreasing beta values
res_betas = [minimize(lambda x: beta_divergence(x, beta), [0, 1], tol=1e-10, method='Nelder-Mead',
                      bounds=[(-100, 100), (0, 100)]) for beta in betas]

f = plt.figure(figsize=(10, 6))
# Create and save figure
x = np.linspace(-4, 8, 1000)
colors = ['m-', 'r-', 'y-']
plt.hist(data, bins=30, density=True, alpha=0.3, color='green', label='Original Distribution')
plt.plot(x, norm.pdf(x, res_kl.x[0], res_kl.x[1]), 'c-', label='Fitted with KL Divergence', lw=4, alpha=0.6)

for i, res_beta in enumerate(res_betas):
    if len(res_betas) == 1:
        plt.plot(x, norm.pdf(x, res_beta.x[0], res_beta.x[1]), colors[i], label=r'Fitted with $\beta$ Divergence', lw=4,
                 alpha=0.6)
    else:
        plt.plot(x, norm.pdf(x, res_beta.x[0], res_beta.x[1]), colors[i], label=f'Fitted with Beta, beta={betas[i]}',
                 lw=4, alpha=0.6)

plt.legend()
plt.xlabel('Value')
plt.ylabel('Density Function')
plt.grid(True)
plt.savefig('divergence_comparison.pdf', format='pdf')
plt.show()
