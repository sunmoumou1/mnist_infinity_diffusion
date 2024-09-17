# 重新复习

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat #type: ignore
from pytorch3d.ops import knn_points, knn_gather #type: ignore

from utils import (
    mean_flat,
    normal_kl,
    discretized_gaussian_log_likelihood,
    ModelMeanType,
    ModelVarType,
    LossType,
    get_conv,
    DCTGaussianBlur,
    _extract_into_tensor
)


class GaussianDiffusion(nn.Module):
    """
    Utilities for training and sampling diffusion models.
    
    :param betas: a 1-D numpy array of betas for each diffusion timestep, starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        gaussian_filter_std=0.0,
        img_size=None, # 要特别注意这里的img_size可能是一个元组
        rescale_timesteps=False,
        multiscale_loss=False,
        multiscale_max_img_size=128, # 要特别注意这里的multiscale_max_img_size可能是一个元组
        mollifier_type="conv", # 还可以选择dct方式
        stochastic_encoding=False,
    ):
        super().__init__()
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        self.multiscale_loss = multiscale_loss
        self.multiscale_max_img_size = multiscale_max_img_size
        self.stochastic_encoding = stochastic_encoding

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        if gaussian_filter_std == 0.0:
            self.mollifier = nn.Identity()
        else:
            if mollifier_type == "conv":
                # 但我们没有使用这种方式
                ksize = math.ceil(gaussian_filter_std * 4 + 1)
                ksize = ksize + 1 if ksize % 2 == 0 else ksize
                self.mollifier = get_conv(
                    (ksize, ksize), (gaussian_filter_std, gaussian_filter_std)
                )
            elif mollifier_type == "dct":
                self.mollifier = DCTGaussianBlur(img_size, gaussian_filter_std)
                

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).
        
        但特别注意这里返回的x_t已经经过了mollification
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return self.mollifier(
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior before T: q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_x_start_component = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
        )
        posterior_x_t_component = (
            _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return (
            posterior_mean,
            posterior_variance,
            posterior_log_variance_clipped,
            posterior_x_start_component,
            posterior_x_t_component,
        )

    def p_mean_variance(
        self, model, x, t, clip_denoised=False, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x_0.

        :param model: the model, which takes a signal and a batch of timesteps as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            # LEARNED：模型直接预测对数方差，并通过指数函数转换为方差。
            # LEARNED_RANGE：模型预测一个范围内的值，通过插值计算出对数方差，再转换为方差。
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = torch.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2 # 这一步转换是因为model_var_values是-1到1之间
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = torch.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)


        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x


        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [
            ModelMeanType.START_X,
            ModelMeanType.EPSILON,
            ModelMeanType.MOLLIFIED_EPSILON,
        ]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                # For ModelMeanType.MOLLIFIED_EPSILON this is actually Tx_0 instead of x_0
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _, posterior_x_start_component, posterior_x_t_component = (
                self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "posterior_x_start_component": posterior_x_start_component,
            "posterior_x_t_component": posterior_x_t_component,
        }


    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            ) * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)


    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
            In particular, cond_fn computes grad(log(p(y|x))), and we want to condition on y.
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ , _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        # todo 怀疑这个地方源代码有错误

        return out

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=False,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        noise_mul=1.0,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = torch.randn_like(x) * noise_mul
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0

        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )

        if self.model_mean_type == ModelMeanType.MOLLIFIED_EPSILON:
            sample = out["mean"] + nonzero_mask * torch.exp(
                0.5 * out["log_variance"]
            ) * self.mollifier(noise)
        else:
            # Otherwise we predict x_0 so need to mollify it.
            sample = self.mollifier(
                out["posterior_x_start_component"]
                + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
            )
            sample = sample + out["posterior_x_t_component"]

        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        noise_mul=1.0,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts(一系列字典), where each dict(每一个字典) is the return value of p_sample().
        """
        if device is None:
            device = next(model.parameters()).device

        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
            img = self.mollifier(img * noise_mul)

        indices = list(range(self.num_timesteps))[::-1]

        if model_kwargs is None:
            model_kwargs = {}

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    noise_mul=noise_mul,
                )
                yield out
                img = out["sample"]

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=False,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        return_all=False,
        noise_mul=1.0,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        all_samples = []
        all_pred_xstarts = []
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            noise_mul=noise_mul,
        ):
            final = sample
            if return_all:
                all_samples.append(sample["sample"][0].float().cpu())
                all_pred_xstarts.append(sample["pred_xstart"][0].float().cpu())
                
        if return_all:
            return (
                final["sample"],
                torch.stack(all_samples),
                torch.stack(all_pred_xstarts),
                final["pred_xstart"],
            )
        else:
            return final["sample"], final["pred_xstart"]


    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        # 这是因为在t=0时，模型的目标是重建原始数据，因此使用 NLL；在其他时间步，模型的目标是匹配后验分布，因此使用 KL 散度。
        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}



    def training_losses(
        self,
        model,
        x_start,
        t,
        encoder=None,
        sample_lst=None,
        model_kwargs=None,
        noise=None,
    ):
        """
        这是最重要的方法
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start:  特别注意这里的x_start是B,C,H,W
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        img_size = (x_start.size(-2), x_start.size(-1))  # 修改：分别取H和W
        x_t = self.q_sample(x_start, t, noise=noise)
        # 注意此时采样得到的这个x_t已经经过了mollification

        mollified_noise = None
        if self.model_mean_type == ModelMeanType.MOLLIFIED_EPSILON:
            mollified_noise = self.mollifier(noise)

        x_start_orig = x_start

        terms = {"x_t": x_t, "noise": noise}  # terms是这个子函数将来要返回的

        if sample_lst is not None:
            # 假设我们有一个 4x4 大小的图像，它被展平成一个 16 个位置的向量。sample_lst 可能是一个包含某些位置的索引的列表，比如 [2, 5, 7]，表示你只对第 2、第 5 和第 7 个像素位置感兴趣。
            # 在这个例子中，如果 sample_lst 是 [2, 5, 7]，那么 torch.gather 将从展平后的图像中提取这三个位置的像素数据。通过在不同通道上重复 sample_lst，你可以确保从这些位置提取完整的多通道数据。
            model_kwargs["sample_lst"] = sample_lst
            x_t = rearrange(x_t, "b c h w -> b (h w) c")
            x_t = torch.gather(
                x_t, 1, sample_lst.unsqueeze(2).repeat(1, 1, x_t.size(2))
            ).contiguous()
            x_start = rearrange(x_start, "b c h w -> b (h w) c")
            x_start = torch.gather(
                x_start, 1, sample_lst.unsqueeze(2).repeat(1, 1, x_start.size(2))
            ).contiguous()
            noise = rearrange(noise, "b c h w -> b (h w) c")
            noise = torch.gather(
                noise, 1, sample_lst.unsqueeze(2).repeat(1, 1, noise.size(2))
            ).contiguous()

            if mollified_noise is not None:
                mollified_noise = rearrange(mollified_noise, "b c h w -> b (h w) c")
                mollified_noise = torch.gather(
                    mollified_noise,
                    1,
                    sample_lst.unsqueeze(2).repeat(1, 1, noise.size(2)),
                ).contiguous()

        if encoder is not None:
            # 这段代码的目的是在训练过程中将输入数据通过一个编码器进行编码，并根据是否使用随机编码（stochastic encoding）来计算相应的损失项（如KL散度）。它还根据给定的索引列表（sample_lst）从输入数据中提取特定位置的数据，并在此基础上进行编码操作。
            sample_lst_fresh = torch.stack(
                [
                    torch.from_numpy(
                        np.random.choice(
                            x_start_orig.size(-2) * x_start_orig.size(-1),  # 修改：使用H*W代替H*H
                            sample_lst.size(1),
                            replace=False,
                        )
                    )
                    for _ in range(sample_lst.size(0))
                ]
            ).to(sample_lst.device) # 此时sample_lst_fresh的形状是B,L
            x_start_orig = rearrange(x_start_orig, "b c h w -> b (h w) c")
            x_start_orig = torch.gather(
                x_start_orig,
                1,
                sample_lst_fresh.unsqueeze(2).repeat(1, 1, x_start_orig.size(2)),
            ).contiguous() # 此时 x_start_orig 形状是B,L,C
            if self.stochastic_encoding:
                encoding, mu, logvar = encoder(
                    x_start_orig, sample_lst=sample_lst_fresh
                )
                kld_loss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1)
                # 其实是比较编码输出的mu和logvar和正态标准正态分布之间的差距
                terms["kld"] = kld_loss
            else: # mark
                encoding = encoder(x_start_orig, sample_lst=sample_lst_fresh)
            model_kwargs["z"] = encoding
            terms["z"] = encoding

        if (self.loss_type == LossType.MSE or self.loss_type == LossType.L1):
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
                ModelMeanType.MOLLIFIED_EPSILON: mollified_noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape

            if self.multiscale_loss:
                terms["multiscale_loss"] = self._multiscale_loss(
                    model_output,
                    target,
                    img_size=img_size,  # 传入(H, W)元组
                    max_img_size=self.multiscale_max_img_size, # 要特别注意这里的 self.multiscale_max_img_size 也是一个元组
                    sample_lst=sample_lst, 
                )

            if self.loss_type == LossType.MSE:
                terms["loss"] = mean_flat((target - model_output) ** 2)
            else:
                terms["loss"] = mean_flat(torch.abs(target - model_output))

            if self.model_mean_type == ModelMeanType.START_X:
                terms["pred_xstart"] = model_output
                
            if "kld" in terms:
                # 注意这一段是自己加上去的，再原先的github代码上并没有这一段
                kld_weight = 1e-4
                terms["loss"] = terms["loss"] + terms["kld"] * kld_weight
                
        else:
            raise NotImplementedError(self.loss_type)

        return terms


    def _multiscale_loss(
        self, model_output, target, img_size=None, max_img_size=None, sample_lst=None
    ):
        # 判断 img_size 和 max_img_size 是否为整数或元组，并分别处理
        if isinstance(img_size, int):
            img_height, img_width = img_size, img_size
        elif isinstance(img_size, tuple) and len(img_size) == 2:
            img_height, img_width = img_size
        else:
            raise ValueError("img_size must be either an integer or a tuple of two integers.")

        if isinstance(max_img_size, int):
            max_img_height, max_img_width = max_img_size, max_img_size
        elif isinstance(max_img_size, tuple) and len(max_img_size) == 2:
            max_img_height, max_img_width = max_img_size
        else:
            raise ValueError("max_img_size must be either an integer or a tuple of two integers.")

        if model_output.ndim == 3 and target.ndim == 3 and sample_lst is not None:
            # Sparse image
            # 1. First calculate loss on the sparse data
            loss = 2 / (img_height + img_width) * torch.mean((target - model_output) ** 2, dim=(1, 2))
            
            # 2. Scatter into max image size
            model_output_grid = self._knn_interpolate_to_grid(
                model_output, sample_lst, (img_height, img_width), (max_img_height, max_img_width)
            )
            target_grid = self._knn_interpolate_to_grid(
                target, sample_lst, (img_height, img_width), (max_img_height, max_img_width)
            )

            # 3. Average pool to each lower size down to a minimum resolution of 32x32
            min_res = (32, 32)
            min_res_height, min_res_width = min_res

            for i in range(1, max(max_img_height // min_res_height, max_img_width // min_res_width)):
                res_height = max_img_height // 2**i
                res_width = max_img_width // 2**i
                model_output_res = F.adaptive_avg_pool2d(model_output_grid, (res_height, res_width))
                target_res = F.adaptive_avg_pool2d(target_grid, (res_height, res_width))
                loss += (
                    2 / (res_height + res_width)
                    * torch.mean((target_res - model_output_res) ** 2, dim=(1, 2, 3))
                )

        elif model_output.ndim == 4 and target.ndim == 4:
            # Full image
            img_height, img_width = model_output.size(-2), model_output.size(-1)
            loss = 0.0
            for i in range(1, max(img_height // 32, img_width // 32)):
                res_height = img_height // 2**i
                res_width = img_width // 2**i
                model_output_res = F.adaptive_avg_pool2d(model_output, (res_height, res_width))
                target_res = F.adaptive_avg_pool2d(target, (res_height, res_width))
                loss += (
                    2 / (res_height + res_width)
                    * torch.mean((target_res - model_output_res) ** 2, dim=(1, 2, 3))
                )

        else:
            raise NotImplementedError("Multiscale loss received unexpected input sizes")

        return loss


    def _knn_interpolate_to_grid(self, x, sample_lst, img_size, max_img_size):
        # 判断 img_size 和 max_img_size 是否为整数或元组
        if isinstance(img_size, int):
            img_height, img_width = img_size, img_size
        elif isinstance(img_size, tuple) and len(img_size) == 2:
            img_height, img_width = img_size
        else:
            raise ValueError("img_size must be either an integer or a tuple of two integers.")

        if isinstance(max_img_size, int):
            max_img_height, max_img_width = max_img_size, max_img_size
        elif isinstance(max_img_size, tuple) and len(max_img_size) == 2:
            max_img_height, max_img_width = max_img_size
        else:
            raise ValueError("max_img_size must be either an integer or a tuple of two integers.")

        # 生成输入网格的坐标
        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(0, 1, steps=img_height), 
                torch.linspace(0, 1, steps=img_width)
            )
        ).to(x.device)
        coords = rearrange(coords, "c h w -> () (h w) c")
        coords = repeat(coords, "() ... -> b ...", b=x.size(0))
        coords = torch.gather(
            coords, 1, sample_lst.unsqueeze(2).repeat(1, 1, coords.size(2))
        ).contiguous()
        # coords 的形状将变成 (B, L, 2)

        # 生成输出网格的坐标
        out_coords = torch.stack(
            torch.meshgrid(
                torch.linspace(0, 1, steps=max_img_height), 
                torch.linspace(0, 1, steps=max_img_width)
            )
        ).to(x.device)
        out_coords = rearrange(out_coords, "c h w -> () (h w) c")

        with torch.no_grad():
            # 计算 KNN 并找到最近的三个邻居
            _, assign_index, neighbour_coords = knn_points(
                out_coords.repeat(x.size(0), 1, 1), coords, K=3, return_nn=True
            )
            # neighbour_coords: (B, y_length, K, 2)
            # y_length 表示的是输出的高分辨率网格中的点的总数，通常等于 max_img_height * max_img_width
            
            # 计算坐标差值
            diff = neighbour_coords - out_coords.unsqueeze(2)
            squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
            weights = 1.0 / torch.clamp(squared_distance, min=1e-16)
            # (B, y_length, K, 1)

        # Inverse square distance weighted mean
        neighbours = knn_gather(x, assign_index)  # (B, y_length, K, C) 注意这里的C可不再是坐标的个数了，而是点属性的向量
        out = (neighbours * weights).sum(2) / weights.sum(2)  # 此时out (B, y_length, C)

        return out.to(x.dtype)



class SpacedDiffusion(GaussianDiffusion):
    """
    SpacedDiffusion 类是 GaussianDiffusion 的一个子类，主要用于在扩散过程中跳过某些时间步，以实现更高效的计算或特殊的扩散过程设计。这个类允许用户选择保留基础扩散过程中的特定时间步，忽略其他时间步，从而自定义扩散过程。

    A diffusion process which can skip steps in a base diffusion process.
    :param use_timesteps: a collection (sequence or set) of timesteps from the original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        
        # Pylint 是一个 Python 的代码分析工具，它可以检查代码中的潜在错误、风格问题以及一些代码质量问题。当一个子类重写父类的方法时，如果重写的方法的参数签名与父类方法的参数签名不匹配，Pylint 会发出 signature-differs 的警告。这种警告表明，子类的方法签名（参数列表）与父类的方法签名不同，这在某些情况下可能导致意外的行为或错误。
        # 在你的代码中，p_mean_variance 方法被重写了，但其签名没有明确指定所有参数，而是使用了不定参数 *args 和 **kwargs。这可能与父类 GaussianDiffusion 中的 p_mean_variance 方法的签名有所不同。因此，Pylint 可能会发出 signature-differs 的警告。
        # 通过添加 # pylint: disable=signature-differs 注释，开发者明确告诉 Pylint 不需要对此发出警告，因为他们有意为之，理解并接受这个不同的签名。这样可以避免不必要的警告干扰代码检查的其他方面。
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
        
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)
