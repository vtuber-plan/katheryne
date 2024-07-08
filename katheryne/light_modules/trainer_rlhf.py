
import trl
import torch
import tqdm
from transformers import PreTrainedModel
from katheryne.utils.hparams import HParams


class TrainerRLHF(object):
    def __init__(self, hparams: HParams, trainer: trl.trainer.BaseTrainer,
                 model: PreTrainedModel, tokenizer,
                 reward_model: PreTrainedModel, reward_tokenizer,
                 ref_model: PreTrainedModel, ref_tokenizer) -> None:
        self.hparams = hparams
        self.trainer = trainer

        self.model, self.tokenizer = model, tokenizer
        self.reward_model, self.reward_tokenizer = reward_model, reward_tokenizer
        self.ref_model, self.ref_tokenizer = ref_model, ref_tokenizer

    def train(self) -> None:
        # Move Reward Model to CUDA
        device = self.trainer.accelerator.device
        if self.trainer.accelerator.num_processes == 1:
            device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
        self.reward_model.to(device)

        # output_length_sampler = LengthSampler(hparams.get("output_min_length", 16), hparams.get("output_max_length", 1024))

        generation_kwargs = {
            "num_beams": 1,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "max_new_tokens": 2048,
        }

        max_epochs = self.hparams.get("max_epochs", 999)
        max_steps = self.hparams.get("max_steps", -1)
        epoch_steps = len(self.trainer.dataloader)

        dataiter = iter(self.trainer.dataloader)

        for epoch in range(max_epochs):
            for step in enumerate(tqdm.tqdm(range(epoch_steps))):
                try:
                    batch = next(dataiter)
                except StopIteration:
                    dataiter = iter(self.trainer.dataloader)
                    batch = next(dataiter)

                # dict_keys(['input_ids', 'attention_mask', 'labels', 'response'])
                query_tensor_input_ids = batch["input_ids"]
                query_tensors = [query_tensor for query_tensor in query_tensor_input_ids]

                response_tensors = self.trainer.generate(query_tensor=query_tensors, batch_size=2, return_prompt=True, **generation_kwargs)

                batch["response"] = [self.tokenizer.decode(r.squeeze()) for r in response_tensors]
                for i in range(len(query_tensors)):
                    print(self.tokenizer.decode(query_tensors[i].squeeze()))
                    print("--------------")
                    print(self.tokenizer.decode(response_tensors[i].squeeze()))
                    print("===========")

                # Compute reward score
                encoded_texts = self.reward_tokenizer(batch["response"],
                    padding="longest",
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=True,
                ).to(self.reward_model.device)

                rewards = self.reward_model.forward(
                    input_ids=encoded_texts["input_ids"],
                    attention_mask=encoded_texts["attention_mask"],
                )
                score_tensor = rewards.logits
                scores = [s.item() for s in score_tensor]
                # Run PPO step
                stats = self.trainer.step(query_tensors, response_tensors, scores)
                self.trainer.log_stats(stats, batch, scores)

                # Save Checkpoints
                # TODO: ....
