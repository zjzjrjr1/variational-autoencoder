import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


batch = 64


# based on the choice, change the cifar10 or mnist
a = 1
if a == 1:
    # Define transformations
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.ToTensor()  # Convert to tensor
    ])
    # Load the CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    dim = 1024

else:
    transform = transforms.Compose([
        transforms.ToTensor()  # Convert images to tensor
         # Normalize with mean 0.5 and std 0.5 for single channel (MNIST is grayscale)
    ])
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    dim = 784

# DataLoader
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch, shuffle=False)       

# Visualize some grayscale images
def imshow(img):
    img = img.squeeze(0)  # Remove channel dimension for grayscale
    plt.imshow(img, cmap='gray')
    plt.axis('off')


class VAE(nn.Module):
    def __init__(self,k, dim ):
        super(VAE, self).__init__()
        self.k = k
        self.encoder_fc1 = nn.Linear(dim, 256)
        self.encoder_mean = nn.Linear(256, self.k)
        self.encoder_var = nn.Linear(256, self.k)
        
        self.decoder_fc1 = nn.Linear(self.k, 256)
        self.decoder_fc2 = nn.Linear(256, dim)
        
    def encoder(self, x):
        x = torch.relu(self.encoder_fc1(x))

        # output the mean and log variance
        # x is in dimension of batch x 64
        mean = torch.relu(self.encoder_mean(x))
        log_variance = torch.relu(self.encoder_var(x))
        return mean, log_variance
    
    def decoder(self, x):
        x = torch.sigmoid(self.decoder_fc1(x))
        x = torch.sigmoid(self.decoder_fc2(x))
        return x
        
    def reparam(self,mu, log_variance):
        eps = torch.normal(0,1,size = (1,self.k))
        if torch.cuda.is_available():
            mu = mu.cuda()
            log_variance = log_variance
            eps = eps.cuda() 
        std = torch.exp(0.5 * log_variance)
        return mu + eps * std
        
    def forward(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        # flatten the x
        x = x.squeeze()
        x = x.view(x.size(0),-1)
        mean, log_variance = self.encoder(x)
        reparam_out = self.reparam(mean, log_variance)
        decoder_out = self.decoder(reparam_out)
        return decoder_out, mean, log_variance
    
    def generate_sample(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        x = x.view(x.size(0),-1)
        mean, log_variance = self.encoder(x)
        reparam_out = self.reparam(mean, log_variance)
        decoder_out = self.decoder(reparam_out)
        return decoder_out, mean, log_variance

# loss function 
def vae_loss(output, target, mean, log_variance, k, beta):
    if torch.cuda.is_available():
        output = output.cuda()
        target = target.cuda()
        mean = mean.cuda()
        log_variance = log_variance.cuda()
    # mean = batch x k
    # log_variance = batch x k
    # kl_dis = batch x 1
    kl_dis = -0.5 * torch.sum(1 + log_variance - mean ** 2 - torch.exp(log_variance), dim=1)
    
    #MSE = batch x (pixel*pixel)
    MSE = 1/2*(output - target)**2 
    # MSE = batch x 1
    MSE = torch.mean(MSE, dim = 1)*dim
    # MSE + kl_dis = batch x 1
    return torch.mean(MSE + kl_dis * beta), torch.mean(MSE), torch.mean(kl_dis)

def train_epoch(VAE_model, train_dataloader, vae_loss,
                optimizer, beta,k):
    epoch_loss = 0
    for train_idx, (features, labels) in enumerate(train_dataloader):
        if torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()
            VAE_model = VAE_model.cuda()
        optimizer.zero_grad()
        # create target
        target = features.view(features.size(0), -1)
        # feed through the encoder
        output, mean, log_variance = VAE_model(features)
        total_loss, MSE_loss, kl_loss = vae_loss(output, target, mean, log_variance, k, beta)
        total_loss.backward()
        optimizer.step()
        # batch loss
        if train_idx % 128 == 0:
            print(f"batch loss is: {total_loss.item()}. MSE loss is {MSE_loss}. kl_loss is {kl_loss}")
        epoch_loss += total_loss.item()  
    return epoch_loss    
    
################# training loop ###############
k = 16
vae_model = VAE(k = k, dim = dim)
vae_model = vae_model.cuda()
epoches = 3
beta = 5e-1 #0.05
sample_num = 5
# start train
vae_model.train()
optimizer = torch.optim.Adam(vae_model.parameters())
for epoch in range(epoches):
    print(f'For epoch of {epoch}')
    epoch_loss = train_epoch(vae_model, train_loader, vae_loss, optimizer, beta, k)

# freeze the model/inference
vae_model.eval()

fig, (axes1, axes2) = plt.subplots(2, sample_num, figsize = (3,3))
for i in range(sample_num): 
    # get the data. 
    image, label = test_dataset[i*10]
    axes2[i].imshow(image.squeeze(0).detach().cpu(), cmap='gray')  # Detach to remove gradients and use cpu if on cuda
    axes2[i].axis('off')  # Turn off axis
    
    generated_image, mu, var = vae_model.generate_sample(image)
    print(mu)
    print(torch.exp(var*0.5))
    # reshape the generated image
    image_samples = generated_image.view(generated_image.size(0), int(dim**0.5),int(dim**0.5))
    axes1[i].imshow(image_samples.squeeze(0).detach().cpu(), cmap='gray')  # Detach to remove gradients and use cpu if on cuda
    axes1[i].axis('off')  # Turn off axis
plt.show()
