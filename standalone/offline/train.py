# finetune_policy_head.py

import torch
import h5py
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ====== PGD Attack Function ======
def pgd_attack(model, criterion, images, labels, device, epsilon, alpha, num_iters):
    original_images = images.clone().detach().to(device)
    adversarial_images = images.clone().detach().to(device)
    
    for _ in range(num_iters):
        adversarial_images.requires_grad_(True)
        outputs = model(adversarial_images)
        loss = criterion(outputs, labels)
            
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            gradient = adversarial_images.grad.data
            adversarial_images = adversarial_images + alpha * gradient.sign()
            
            perturbations = torch.clamp(adversarial_images - original_images, min=-epsilon, max=epsilon)
            adversarial_images = torch.clamp(original_images + perturbations, 0, 1)
        
        adversarial_images = adversarial_images.detach()
    return adversarial_images, perturbations


# ====== Custom Dataset for HDF5 Loading ======
class FusedFeatureDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_file = h5py.File(h5_path, 'r')
        self.features = self.h5_file['features']
        self.supervision = self.h5_file['supervision']
        print(f"[INFO] Loaded dataset with {len(self.features)} samples from {h5_path}")


    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.supervision[idx], dtype=torch.float32)
        return x, y


# ====== Model Wrapper for Policy Head Only ======
class HeadOnlyPolicy(nn.Module):
    def __init__(self, mlp_head):
        super().__init__()
        self.mlp_head = mlp_head

    def forward(self, fused_input):
        return self.mlp_head(fused_input)


# ====== Load Original Policy Model (Customize this part to match your structure) ======
def load_full_policy_model(path):
    from standalone.rsl_rl.ext.modules import VisionActorCritic  # Replace with your actual module/class
    model = VisionActorCritic(
        num_actor_obs=6928,  # Adjust based on your feature size
        num_critic_obs=6928,  # Adjust based on your action space
        num_actions=4,
        img_res=(72, 96),  # Adjust based on your input image resolution
        dim_hidden_input=192,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        activation='lrelu',
        noise_std_type='scalar',
        init_noise_std=1.0,
        use_auxiliary_loss=True  # Set to True if your model uses an auxiliary head
    )
    loaded_dict = torch.load(path, weights_only=False)
    model.load_state_dict(loaded_dict['model_state_dict'])
    return loaded_dict, model


# ====== Main Fine-tuning Procedure ======
def finetune_policy_head(h5_path, policy_path, save_path, epochs=10, batch_size=256, lr=3e-4):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Dataset
    dataset = FusedFeatureDataset(h5_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Load pretrained policy and extract mlp_head
    loaded_dict, full_policy = load_full_policy_model(policy_path)
    head_model = HeadOnlyPolicy(full_policy.aux_decoder).to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(head_model.parameters(), lr=lr)

    # Training loop
    head_model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            x, _ = pgd_attack(head_model, criterion, x, y, device, epsilon=0.1, alpha=0.01, num_iters=3)

            pred = head_model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataset)
        print(f"\n🚀 [Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.6f}")

        # Save updated full policy
        full_policy.aux_decoder.load_state_dict(head_model.mlp_head.state_dict())
        loaded_dict["model_state_dict"] = full_policy.state_dict()
        torch.save(loaded_dict, save_path + f"/policy_finetune_epoch-{epoch}.pt")
        print(f"\n✅ Fine-tuned policy saved to: {save_path}")


# ====== Entry Point ======
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune policy MLP head using HDF5 offline data")
    parser.add_argument('--h5_path', type=str, required=True, help='Path to HDF5 dataset')
    parser.add_argument('--policy_path', type=str, required=True, help='Path to pretrained policy .pth file')
    parser.add_argument('--save_path', type=str, default='policy_finetuned.pth', help='Where to save fine-tuned policy')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)

    args = parser.parse_args()
    finetune_policy_head(args.h5_path, args.policy_path, args.save_path, args.epochs, args.batch_size, args.lr)