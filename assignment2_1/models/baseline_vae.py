import torch
from torch import device, nn
from torch.nn import functional as F

class BaselineVAE(nn.Module):
    def __init__(self,in_channels,hidden_dims,latent_dim,device='cuda'):
        super().__init__()
        self.hidden_dims=hidden_dims
        self.latent_dim=latent_dim
        self.device = device
        self.in_channels=in_channels
        self.encoder=self._build_encoder()
        self.decoder=self._build_decoder()
        # self.kld_weight=kld_weight
        with torch.no_grad():
            dummy_output = self.encoder(torch.ones(1,in_channels,128,128))
            flatten_dim = dummy_output.view(1, -1).size(1)
        self.fc_mu=nn.Linear(flatten_dim,self.latent_dim)
        self.fc_logvar=nn.Linear(flatten_dim,self.latent_dim)
        self.decoder_input = nn.Linear(self.latent_dim,flatten_dim)
        self.final_layer = self._build_final_layer()        

    def _build_encoder(self):
        layers=[]
        in_channels = self.in_channels
        for hidden_dim in self.hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels,out_channels=hidden_dim,kernel_size=3,stride=2,padding=1),
                    nn.BatchNorm2d(hidden_dim),     
                    nn.LeakyReLU())
            )
            in_channels = hidden_dim
        # layers.append(nn.Linear(in_channels,self.latent_dim))
        return nn.Sequential(*layers)



    def _build_decoder(self):
        layers=[]
        reversed_hidden_dims = self.hidden_dims[::-1]
        for i in range(len(reversed_hidden_dims)-1):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(reversed_hidden_dims[i],out_channels=reversed_hidden_dims[i+1],kernel_size=3,stride=2,padding=1,output_padding=1),
                    nn.BatchNorm2d(reversed_hidden_dims[i+1]),     
                    nn.LeakyReLU()
                )   
            )    
        # Add an additional upsampling layer to match the encoder downsampling depth
        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(reversed_hidden_dims[-1],out_channels=reversed_hidden_dims[-1],kernel_size=3,stride=2,padding=1,output_padding=1),
                nn.BatchNorm2d(reversed_hidden_dims[-1]),
                nn.LeakyReLU()
            )
        )
        return nn.Sequential(*layers)

    def _build_final_layer(self):
        reversed_hidden_dims = self.hidden_dims[::-1]
        return nn.Sequential(
                    nn.Conv2d(in_channels=reversed_hidden_dims[-1],out_channels=reversed_hidden_dims[-1],kernel_size=3,stride=1,padding=1),
                    nn.BatchNorm2d(reversed_hidden_dims[-1]),
                    nn.LeakyReLU(),
                    nn.Conv2d(in_channels=reversed_hidden_dims[-1],out_channels=self.in_channels,kernel_size=3,stride=1,padding=1),                
                    nn.Tanh()
                )    

    def reparameterize(self,mu,logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def sample(self,num_samples,current_device):
        z = torch.randn(num_samples,self.latent_dim).to(current_device)
        x = self.decoder_input(z)
        h = int((self.fc_mu.in_features // self.hidden_dims[-1]) ** 0.5)
        x = x.view(num_samples,self.hidden_dims[-1],h,h)
        x = self.decoder(x)
        return self.final_layer(x)

    def forward(self,x):
        x= self.encoder(x)
        batch_size,_,h,w = x.size()
        x= x.view(batch_size, -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu,logvar)
        x = self.decoder_input(z)
        x = x.view(batch_size,self.hidden_dims[-1],h,w)
        x = self.decoder(x)
        return self.final_layer(x), mu, logvar

    def loss(self,x,kld_weight):
        # x=x.to(self.device)
        x_recon,mu,logvar = self.forward(x)
        batch_size = x.size(0)
        recon_loss = F.mse_loss(x_recon, x,reduction = 'mean')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())/batch_size
        return recon_loss + kld_weight * kld_loss , recon_loss, kld_loss
    
    def generate(self,x,**kwargs):
        return self.forward(x)[0]