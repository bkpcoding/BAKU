import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
import numpy as np
import utils


class BCLightningModule(pl.LightningModule):
    def __init__(
        self,
        agent,
        expert_dataset,
        batch_size,
        stats,
        cfg
    ):
        super().__init__()
        self.automatic_optimization = False
        # Register the agent as a module instead of just storing it
        self.agent = self._register_agent(agent)
        
        self.expert_dataset = expert_dataset
        self.batch_size = batch_size
        self.stats = stats
        self.cfg = cfg
        
        # Save hyperparameters for logging and checkpointing
        self.save_hyperparameters(ignore=["agent", "expert_dataset"])
        
        # Variables that used to be in WorkspaceIL
        self._global_step = 0
        self._global_episode = 0
        
    def _register_agent(self, agent):
        """Helper method to properly register the agent's components as modules"""
        # Register main components
        if hasattr(agent, 'actor'):
            self.register_module('actor', agent.actor)
        if hasattr(agent, 'encoder'):
            if isinstance(agent.encoder, dict):
                # For separate encoders
                for key, encoder in agent.encoder.items():
                    self.register_module(f'encoder_{key}', encoder)
            else:
                self.register_module('encoder', agent.encoder)
        if hasattr(agent, 'proprio_projector'):
            self.register_module('proprio_projector', agent.proprio_projector)
        if hasattr(agent, 'language_projector'):
            self.register_module('language_projector', agent.language_projector)
            
        # Verify that we have trainable parameters after registration
        has_params = False
        for param in self.parameters():
            if param.requires_grad:
                has_params = True
                break
                
        if not has_params:
            raise ValueError("Model has no trainable parameters after registration! Check agent components.")
            
        return agent

    def forward(self, batch):
        # Forward pass logic
        # This will be used for inference/validation
        data = utils.to_torch(batch, self.device)
        action = data["actions"].float()
        
        # Get metrics/loss but don't update model
        metrics = self.agent.compute_loss(batch, self.global_step)
        return metrics
    
    def training_step(self, batch, batch_idx):
        # Get all optimizers
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]

        # Zero all gradients
        for opt in optimizers:
            opt.zero_grad()

        # Compute loss and metrics
        metrics = self.agent.compute_loss(batch, self.global_step)
        loss = metrics["actor_loss"]
        
        # Backward pass
        self.manual_backward(loss)
        
        # Step all optimizers
        for opt in optimizers:
            opt.step()
        
        self._global_step += 1
        
        # Log metrics
        for key, value in metrics.items():
            self.log(f"train/{key}", value, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return metrics["actor_loss"]
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Validation step logic
        metrics = self.agent.compute_loss(batch, self.global_step)
        
        # Log validation metrics
        self.log("val/actor_loss", metrics["actor_loss"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return metrics
    
    def on_validation_epoch_end(self):
        # Check if we have per-skill validation data
        if hasattr(self.expert_dataset, 'get_validation_batch_per_skill'):
            # Get per-skill validation batches
            val_batches_per_skill = self.expert_dataset.get_validation_batch_per_skill(
                batch_size_per_skill=self.cfg.suite.get('validation_batch_size', 32)
            )
            
            # Process each skill separately
            skill_losses = {}
            for skill_name, val_batches in val_batches_per_skill.items():
                skill_loss = self._validate_skill(skill_name, val_batches)
                
                if skill_loss is not None:
                    skill_losses[skill_name] = skill_loss
                    self.log(f"val/actor_loss_{skill_name}", skill_loss, sync_dist=True)
                    
                    # Update best skill loss
                    if skill_loss < self.best_val_losses[skill_name]:
                        self.best_val_losses[skill_name] = skill_loss
                        self.log(f"val/actor_loss_{skill_name}_best", skill_loss, sync_dist=True)
            
            # Compute overall validation loss
            if skill_losses:
                overall_val_loss = sum(skill_losses.values()) / len(skill_losses)
                self.log("val/actor_loss_overall", overall_val_loss, sync_dist=True)
            
        # For standard validation without skills, the metrics are already logged in validation_step
    
    def _validate_skill(self, skill_name, val_batches):
        """Process validation batches for a specific skill"""
        from collections import defaultdict
        import torch
        
        total_skill_loss = 0
        num_valid_samples = 0
        
        # Collect all samples for this skill into a single batch
        batch_data = defaultdict(list)
        for val_batch in val_batches:
            for k, v in val_batch.items():
                batch_data[k].append(v)
        
        # Convert lists to appropriate format
        processed_batch = {}
        for k, v in batch_data.items():
            if isinstance(v[0], (np.ndarray, torch.Tensor)):
                processed_batch[k] = np.stack(v, axis=0)
            else:
                processed_batch[k] = v
        
        try:
            metrics = self.agent.compute_loss(processed_batch, self.global_step)
            total_skill_loss += metrics['actor_loss']
            num_valid_samples += 1
        except Exception as e:
            print(f"Warning: Error processing validation batch for skill {skill_name}: {str(e)}")
            return None
        
        if num_valid_samples > 0:
            return total_skill_loss / num_valid_samples
        else:
            return None
    
    def configure_optimizers(self):
        # Lightning will handle the optimizer step calls
        # Just return the optimizer(s) that you already created in your agent
        optimizers = []
        
        # Since the agent already initializes its optimizers, we'll just extract them
        # Add actor optimizer(s)
        if isinstance(self.agent.actor_opt, list):
            optimizers.extend(self.agent.actor_opt)
        else:
            optimizers.append(self.agent.actor_opt)
        
        # Add encoder optimizer if applicable
        if hasattr(self.agent, 'encoder_opt') and self.agent.train_encoder:
            optimizers.append(self.agent.encoder_opt)
            
        # Add proprio optimizer if applicable
        if hasattr(self.agent, 'proprio_opt') and self.agent.use_proprio:
            optimizers.append(self.agent.proprio_opt)
            
        # Add language optimizer if applicable
        if hasattr(self.agent, 'language_opt') and self.agent.use_language:
            optimizers.append(self.agent.language_opt)
            
        return optimizers
    
    def train_dataloader(self):
        # Create a DataLoader that yields random batches from the dataset
        # For BC datasets that are IterableDataset, we use appropriate batch size
        # Note: for DDP, Lightning takes care of properly sharding the data
        return DataLoader(
            self.expert_dataset,
            batch_size=self.batch_size, 
            num_workers=self.cfg.get('num_workers', 4),
            pin_memory=self.cfg.get('pin_memory', True),
            persistent_workers=self.cfg.get('persistent_workers', True),
        )
    
    def val_dataloader(self):
        # If there's a specific validation set, use it; otherwise use train data
        if hasattr(self.expert_dataset, 'get_validation_dataset'):
            val_dataset = self.expert_dataset.get_validation_dataset()
            return DataLoader(
                val_dataset,
                batch_size=self.cfg.get('validation_batch_size', 32),
                num_workers=self.cfg.get('num_workers', 4),
                pin_memory=self.cfg.get('pin_memory', True),
                persistent_workers=self.cfg.get('persistent_workers', True),
            )
        else:
            # Use the same dataset but in validation mode
            return DataLoader(
                self.expert_dataset,
                batch_size=self.cfg.get('validation_batch_size', 32),
                num_workers=self.cfg.get('num_workers', 4),
                pin_memory=self.cfg.get('pin_memory', True),
                persistent_workers=self.cfg.get('persistent_workers', True),
            )
    
    @property
    def global_step(self):
        # For compatibility with existing code
        return self._global_step
    
    @property
    def global_episode(self):
        return self._global_episode
    
    @property
    def global_frame(self):
        return self.global_step * self.cfg.suite.action_repeat