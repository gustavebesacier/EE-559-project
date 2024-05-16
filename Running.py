from tqdm.auto import tqdm

train_cycle_with_distillation(target_models[target], teacher_model,tokenizer, optimizer,
                                                                            criterion, metrics, train_loader,
                                                                            test_loader, n_epochs=num_epochs, device=device,
                                                                            alpha=0.25, T=2)

tokenized_imdb = tokenizer(examples["text"], padding="max_length", truncation=True)
tokenized_imdb.set_format("torch")
small_train_dataset = tokenized_imdb["train"].shuffle(seed=42).select(range(1000))
train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
inputs = tokenizer.batch_encode_plus(x_batch, return_tensors="pt", padding=True, truncation=True)
def trainer(model, optimizer, train_loader, nbr_epochs, device):
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(n_epochs):
        for batch_num, (tokens, labels) in tqdm(enumerate(train_loader)):
            #assume batchmaker is right
            model_outputs = model(**tokens, labels=labels)

            loss = model_outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()

            #clean optimizer
            optimizer.zero_grad()

            progress_bar.update(1)



# iterate over epochs
for epoch in range(num_epochs):
    # iterate over batches in training set
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model_bert_l4(**batch)
        # 1 line of code
        ### BEGIN SOLUTION
        loss = outputs.loss
        ### END SOLUTION

        # do the backward pass
        # 1 line of code
        ### BEGIN SOLUTION
        loss.backward()
        ### END SOLUTION

        # perform one step of the optimizer
        # 1 line fo code
        ### BEGIN SOLUTION
        optimizer.step()
        ### END SOLUTION

        # peform one step of the lr_scheduler, similar with the optimizer
        # 1 line of code
        ### BEGIN SOLUTION
        lr_scheduler.step()
        ### END SOLUTION

        # zero the gradients, call zero_grad() on the optimizer
        # 1 line of code
        ### BEGIN SOLUTION
        optimizer.zero_grad()
        ### END SOLUTION

        progress_bar.update(1)