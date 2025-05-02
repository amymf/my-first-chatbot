import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import train_dataset, val_dataset
import model

wandb.init(project="my-first-chatbot")

torch.manual_seed(42)  # For reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

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
    train_loss = 0
    for batch in train_dataloader:
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

        train_loss += loss.item()
        
    # Validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(device)
            response_ids = batch["response_ids"].to(device)
            input_padding_mask = batch["input_padding_mask"].to(device)
            response_padding_mask = batch["response_padding_mask"].to(device)

            # PyTorch expects False for real tokens, True for padding tokens
            input_padding_mask = ~batch["input_padding_mask"].to(device)
            response_padding_mask = ~batch["response_padding_mask"].to(device)

            output = model(input_ids, response_ids, input_padding_mask, response_padding_mask)
            loss = criterion(output.view(-1, output.size(-1)), response_ids.view(-1))
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_dataloader)
    avg_train_loss = train_loss / len(train_dataloader)
    wandb.log({
        "epoch": epoch,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss
    })
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")