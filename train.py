import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import train_dataset
import model

wandb.init(project="my-first-chatbot")

torch.manual_seed(42)  # For reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = model.Transformer(
    embed_dim=512, # defined by tokenizer
    max_len=128,
    num_heads=8,
    num_layers=6,
    vocab_size=train_dataset.tokenizer.vocab_size
).to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.tokenizer.pad_token_id)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        response_ids = batch["response_ids"].to(device)
        input_padding_mask = batch["input_padding_mask"].to(device)
        response_padding_mask = batch["response_padding_mask"].to(device)
        
        # PyTorch expects False for real tokens, True for padding tokens
        input_padding_mask = ~batch["input_padding_mask"].to(device)
        response_padding_mask = ~batch["response_padding_mask"].to(device)

        optimizer.zero_grad()
        output = model(input_ids, response_ids, input_padding_mask, response_padding_mask)
        loss = criterion(output.view(-1, output.size(-1)), response_ids.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    wandb.log({"epoch": epoch, "loss": avg_loss})
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")