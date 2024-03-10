"""Models available in pytorch-generative."""

from src.pytorch_generative.pytorch_generative.models.autoregressive.fvbn import (
    FullyVisibleBeliefNetwork,
)
from src.pytorch_generative.pytorch_generative.models.autoregressive.gated_pixel_cnn import (
    GatedPixelCNN,
)
from src.pytorch_generative.pytorch_generative.models.autoregressive.image_gpt import (
    ImageGPT,
)
from src.pytorch_generative.pytorch_generative.models.autoregressive.made import MADE
from src.pytorch_generative.pytorch_generative.models.autoregressive.nade import NADE
from src.pytorch_generative.pytorch_generative.models.autoregressive.pixel_cnn import (
    PixelCNN,
)
from src.pytorch_generative.pytorch_generative.models.autoregressive.pixel_snail import (
    PixelSNAIL,
)
from src.pytorch_generative.pytorch_generative.models.flow.nice import NICE
from src.pytorch_generative.pytorch_generative.models.kde import (
    GaussianKernel,
    KernelDensityEstimator,
    ParzenWindowKernel,
)
from src.pytorch_generative.pytorch_generative.models.mixture_models import (
    BernoulliMixtureModel,
    GaussianMixtureModel,
)
from src.pytorch_generative.pytorch_generative.models.vae.beta_vae import BetaVAE
from src.pytorch_generative.pytorch_generative.models.vae.vae import VAE
from src.pytorch_generative.pytorch_generative.models.vae.vd_vae import VeryDeepVAE
from src.pytorch_generative.pytorch_generative.models.vae.vq_vae import (
    VectorQuantizedVAE,
)
from src.pytorch_generative.pytorch_generative.models.vae.vq_vae_2 import (
    VectorQuantizedVAE2,
)
