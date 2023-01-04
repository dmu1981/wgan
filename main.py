"""Wasserstein GAN"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
import torchvision
from tqdm import tqdm

device = "cuda"
LATENT_DIM = 100
IMAGE_SIZE = 64
BATCH_SIZE = 256

def weights_init(m):
    """Helper for weight initialization. 
     Weights are initialized very close to zero to avoid being clipped right away"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.005)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.005)
        nn.init.constant_(m.bias.data, 0)

class Down(nn.Module):
    """Strided down convolution, resolution is halved"""
    def __init__(self, in_channels, out_channels, bn):
        super().__init__()
        if bn:
            self.fw = nn.Sequential(
              nn.Conv2d(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(4,4),
                        stride=(2,2),
                        padding=1,
                        bias=False),
              nn.BatchNorm2d(num_features=out_channels),
              nn.LeakyReLU(0.2)
            )
        else:
            self.fw = nn.Sequential(
              nn.Conv2d(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(4,4),
                        stride=(2,2),
                        padding=1,
                        bias=False),
              nn.LeakyReLU(0.2)
            )

    def forward(self, x):
        return self.fw(x)

class Up(nn.Module):
    """Strided up convolution, resolution is doubled"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fw = nn.Sequential(
          nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=4, stride=2, padding=1, bias=False),
          nn.BatchNorm2d(num_features=out_channels),
          nn.ReLU()
        )

    def forward(self, x):
        return self.fw(x)

class Discriminator(nn.Module):
    """The discrimnator (critic)"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
          Down(  3,  64, False),
          Down( 64, 128, True),
          Down(128, 256, True),
          Down(256, 512, True),
          nn.Conv2d(512, 256,4,1,0),
          nn.Flatten(),
        )

    def forward(self, x):
        return self.encoder(x)

class Generator(nn.Module):
    """The generator network"""
    def __init__(self):
        super().__init__()
        self.upconv = nn.Sequential(
          nn.ConvTranspose2d(LATENT_DIM, 512, 4, 1, 0, bias=False),
          nn.BatchNorm2d(512),
          nn.ReLU(),
          Up(512, 256),
          Up(256,128),
          Up(128, 64),
          nn.ConvTranspose2d(64, 3, 4, 2, 1),
          nn.Tanh()
        )

    def forward(self, x):
        return self.upconv(x.view(-1,LATENT_DIM,1,1))

def get_data(dataset_path, batch_size):
    """Builds the dataloader for our samples"""
    BORDER = IMAGE_SIZE // 8
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(
          (IMAGE_SIZE+BORDER,IMAGE_SIZE+BORDER),
          interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        torchvision.transforms.RandomRotation(
          15, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
        torchvision.transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
        torchvision.transforms.Lambda(lambda x: x.to(device))
    ])
    dataset = torchvision.datasets.ImageFolder(dataset_path, transform=transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

dataloader = get_data("./train", BATCH_SIZE)
GAN_PATH = "gan.pt"

class GANTrainer:
    """ The training class for the Wasserstein GAN"""
    def __init__(self):
        self.generator = Generator().to(device)
        self.discriminator = Discriminator().to(device)

        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

        self.generator_optimizer = optim.RMSprop(self.generator.parameters(), lr=0.00005)
        self.discriminator_optimizer = optim.RMSprop(self.discriminator.parameters(),lr=0.00005)

        self.dist = torch.distributions.Normal(0, 1)
        self.dist.loc = self.dist.loc.cuda()
        self.dist.scale = self.dist.scale.cuda()

        self.cnt_g = 0
        self.cnt_d = 0
        self.loss_d = 0
        self.loss_g = 0

        self.one_labels = torch.tensor([1]).tile(BATCH_SIZE)\
            .type(torch.float).to(device).view(BATCH_SIZE,1)

        self.zero_labels = torch.tensor([0]).tile(BATCH_SIZE)\
            .type(torch.float).to(device).view(BATCH_SIZE,1)

        self.start_epoch = 0
        try:
            checkpoint = torch.load("gan.pt")

            self.generator.\
              load_state_dict(checkpoint['generator_model_state_dict'])
            self.generator_optimizer.\
              load_state_dict(checkpoint['generator_optimizer_state_dict'])
            self.discriminator.\
              load_state_dict(checkpoint['discriminator_model_state_dict'])
            self.discriminator_optimizer.\
              load_state_dict(checkpoint['discriminator_optimizer_state_dict'])

            self.start_epoch = checkpoint['epoch']
        except:
            print("Could not load model from disk, starting from scratch")

    def train_discriminator(self, epoch, bar, it):
        """Training step for the discriminator"""
        for _ in range(2):
            batch, _ = next(it)

            if batch.size(0) != BATCH_SIZE:
                it = iter(dataloader)
                batch, _ = next(it)

            # Reset gradients
            self.discriminator_optimizer.zero_grad()

            # All real batch
            x_real = self.discriminator(batch)

            z = self.dist.sample((BATCH_SIZE, LATENT_DIM))
            generated_images = self.generator(z)

            # Run through the discriminator
            x_fake = self.discriminator(generated_images)

            loss_d = torch.mean(x_fake) - torch.mean(x_real)
            loss_d.backward()
            self.loss_d += loss_d.item()
            self.cnt_d += 1

            # Do optimization step
            self.discriminator_optimizer.step()

            # Wasserstein GAN requires to clip weighs in the discriminator
            # to constrain the otherwise unconstrained loss
            for p in self.discriminator.parameters(True):
                p.data.clamp_(-0.01, 0.01)

            if self.cnt_d > 0 and self.cnt_g > 0:
                bar.set_description("G: epoch {}, ld={:.3f}, lg={:.3f}"\
                   .format(epoch, self.loss_d / self.cnt_d, self.loss_g / self.cnt_g))

        return it

    def train_generator(self, epoch, bar):
        """Training step for the generator"""
        for _ in range(1):
            self.generator_optimizer.zero_grad()

            # Sample z for a full batch
            z = self.dist.sample((BATCH_SIZE, LATENT_DIM))
            generated_images = self.generator(z)

            # Run through the discriminator
            x = self.discriminator(generated_images)
            loss = -torch.mean(x)

            # Backpropagation
            loss.backward()
            self.generator_optimizer.step()

            self.loss_g += loss.item()

            self.cnt_g += 1

            if self.cnt_d > 0 and self.cnt_g > 0:
                bar.set_description("G: epoch {}, ld={:.3f}, lg={:.3f}"\
                    .format(epoch, self.loss_d / self.cnt_d, self.loss_g / self.cnt_g))

    def save_images(self, epoch):
        """Generate some images and save them to disk for review"""
        z = self.dist.sample((64, LATENT_DIM))
        with torch.no_grad():
            generated_images = self.generator(z)
        img = (generated_images + 1) / 2

        grid = torchvision.utils.make_grid(img, nrow=8)
        im = torchvision.transforms.ToPILImage()(grid)
        im.save("epoch_{}.png".format(epoch))

    def train(self):
        """Train some epochs"""
        it = iter(dataloader)
        for epoch in range(self.start_epoch, 25000):
            self.loss_d = 0
            self.loss_g = 0
            self.cnt_d = 0
            self.cnt_g = 0

            bar = tqdm(range(20))
            for _ in bar:
                it = self.train_discriminator(epoch, bar, it)
                self.train_generator(epoch, bar)

            self.save_images(epoch)

            torch.save({
                      'epoch': epoch,
                      'generator_model_state_dict': 
                        self.generator.state_dict(),
                      'generator_optimizer_state_dict': 
                        self.generator_optimizer.state_dict(),
                      'discriminator_model_state_dict': 
                        self.discriminator.state_dict(),
                      'discriminator_optimizer_state_dict': 
                        self.discriminator_optimizer.state_dict(),
                      }, "gan.pt")

Trainer = GANTrainer()
Trainer.train()
