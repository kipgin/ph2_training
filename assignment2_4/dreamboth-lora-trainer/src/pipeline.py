import torch
import torch.nn.functional as F

class TrainingPipeline:
    """
    Step execution layer that abstracts away the forward diffusion process,
    latent encoding, text embedding, and loss calculation for a training iteration.
    """
    def __init__(
        self,
        accelerator,
        unet,
        text_encoder,
        vae,
        noise_scheduler,
        optimizer,
        lr_scheduler,
        with_prior_preservation=False,
        prior_loss_weight=1.0
    ):
        self.accelerator = accelerator
        self.unet = unet
        self.text_encoder = text_encoder
        self.vae = vae
        self.noise_scheduler = noise_scheduler
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.with_prior_preservation = with_prior_preservation
        self.prior_loss_weight = prior_loss_weight

    def training_step(self, batch):
        """
        Executes a single forward training step:
        1. Encodes target images to latent space via the VAE.
        2. Samples random noise and timesteps.
        3. Adds noise to the latents (forward diffusion).
        4. Encodes prompt input IDs via the text encoder.
        5. Predicts noise via the U-Net.
        6. Computes MSE loss (optionally with class regularization).
        
        Returns:
            loss (Tensor): Total loss scalar.
            loss_instance (Tensor): Instance image loss component.
            loss_prior (Tensor or None): Prior/class regularization loss, or None if disabled.
        """
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        device = pixel_values.device
        # Use the VAE's dtype as the reference — it is always frozen and cast to the
        # exact target precision in train.py. Sniffing from self.unet would return
        # float32 because LoRA parameters are kept in fp32 for gradient accuracy even
        # during fp16/bf16 mixed-precision training, causing a dtype mismatch with the
        # fp16 VAE conv layers.
        weight_dtype = next(self.vae.parameters()).dtype

        # Cast pixel values to the precision of the model
        pixel_values = pixel_values.to(dtype=weight_dtype)

        # 1. Encode images into latent space via VAE
        # Standard SD scaling factor is applied to normalize the latent distribution
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # 2. Sample random noise of the same shape
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        # 3. Sample random timesteps
        num_train_timesteps = getattr(self.noise_scheduler.config, "num_train_timesteps", 1000)
        timesteps = torch.randint(0, num_train_timesteps, (bsz,), device=device).long()

        # 4. Add noise to latents (forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # 5. Encode text inputs through the text encoder
        encoder_hidden_states = self.text_encoder(input_ids)[0]

        # 6. Predict noise via U-Net
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample

        # 7. Calculate loss (MSE loss)
        if self.with_prior_preservation:
            # Under prior preservation, batch is stacked: first half is instance, second half is class
            noise_pred_instance, noise_pred_class = noise_pred.chunk(2, dim=0)
            noise_instance, noise_class = noise.chunk(2, dim=0)

            loss_instance = F.mse_loss(noise_pred_instance.float(), noise_instance.float(), reduction="mean")
            loss_prior = F.mse_loss(noise_pred_class.float(), noise_class.float(), reduction="mean")

            loss = loss_instance + self.prior_loss_weight * loss_prior
        else:
            loss_instance = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            loss_prior = None
            loss = loss_instance

        return loss, loss_instance, loss_prior
