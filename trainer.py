import torch
import torch.nn as nn
import copy
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_recall_curve, precision_score, f1_score as sk_f1_score
from models import binary_cross_entropy, cross_entropy_logits, entropy_logits, RandomLayer
from prettytable import PrettyTable
from domain_adaptator import ReverseLayerF
from tqdm import tqdm


class Trainer(object):
    def _to_device(self, data):
        """Helper method to move data to device, handling tuples, tensors, and DGL graphs"""
        if isinstance(data, tuple):
            return tuple(item.to(self.device) for item in data)
        elif hasattr(data, 'to') and hasattr(data, 'ndata'):
            # DGL graph - move to device
            return data.to(self.device)
        elif hasattr(data, 'to'):
            # Tensor
            return data.to(self.device)
        else:
            # Already on device or doesn't need moving
            return data

    def __init__(self, model, optim, device, train_dataloader, val_dataloader, test_dataloader, opt_da=None, discriminator=None,
                 experiment=None, alpha=1, **config):
        self.model = model
        self.optim = optim
        self.device = device
        self.epochs = config["SOLVER"]["MAX_EPOCH"]
        self.current_epoch = 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.is_da = config["DA"]["USE"]
        self.alpha = alpha
        self.n_class = config["DECODER"]["BINARY"]
        self.config = config  # Store full config for support set checking
        self.optim_da = opt_da  # Always set, can be None for MMD
        if self.is_da:
            self.da_method = config["DA"]["METHOD"]
            self.domain_dmm = discriminator
            if config["DA"]["RANDOM_LAYER"] and not config["DA"]["ORIGINAL_RANDOM"]:
                self.random_layer = nn.Linear(in_features=config["DECODER"]["IN_DIM"]*self.n_class, out_features=config["DA"]
                ["RANDOM_DIM"], bias=False).to(self.device)
                torch.nn.init.normal_(self.random_layer.weight, mean=0, std=1)
                for param in self.random_layer.parameters():
                    param.requires_grad = False
            elif config["DA"]["RANDOM_LAYER"] and config["DA"]["ORIGINAL_RANDOM"]:
                self.random_layer = RandomLayer([config["DECODER"]["IN_DIM"], self.n_class], config["DA"]["RANDOM_DIM"])
                if torch.cuda.is_available():
                    self.random_layer.cuda()
            else:
                self.random_layer = False
        self.da_init_epoch = config["DA"]["INIT_EPOCH"]
        self.init_lamb_da = config["DA"]["LAMB_DA"]
        self.batch_size = config["SOLVER"]["BATCH_SIZE"]
        self.use_da_entropy = config["DA"]["USE_ENTROPY"]
        self.nb_training = len(self.train_dataloader)
        self.step = 0
        self.experiment = experiment

        self.best_model = None
        self.best_epoch = None
        self.best_auroc = 0
        self.best_val_loss = float('inf')  # For multitask: use val loss as criterion

        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.train_da_loss_epoch = []
        self.val_loss_epoch, self.val_auroc_epoch = [], []
        self.test_metrics = {}
        self.config = config
        self.output_dir = config["RESULT"]["OUTPUT_DIR"]

        # Knowledge Distillation settings
        self.use_distillation = config.get("DISTILLATION", {}).get("ENABLED", False)
        if self.use_distillation:
            self.distill_temperature = config["DISTILLATION"].get("TEMPERATURE", 3.0)
            self.distill_alpha = config["DISTILLATION"].get("ALPHA_DISTILL", 0.5)
            self.expert_low_model = None
            self.expert_high_model = None
            print(f"Knowledge Distillation Enabled: T={self.distill_temperature}, alpha={self.distill_alpha}")

        # Multi-task learning configuration
        self.use_multitask = config.get("MULTITASK", {}).get("ENABLED", False)
        if self.use_multitask:
            self.multitask_alpha = config["MULTITASK"].get("ALPHA", 0.5)  # Classification loss weight
            self.multitask_beta = config["MULTITASK"].get("BETA", 0.5)   # Regression loss weight
            print(f"Multitask Learning Enabled: α={self.multitask_alpha} (classification), β={self.multitask_beta} (regression)")

        # Add R² to headers if using multitask learning
        if self.n_class >= 3:
            # Multi-class classification headers (3, 10, or any number of classes)
            if self.use_multitask:
                valid_metric_header = ["# Epoch", "AUROC", "F1_weighted", "Val_loss", "R²"]
                test_metric_header = ["# Best Epoch", "AUROC", "F1_weighted",
                                      "Accuracy", "Test_loss", "R²"]
            else:
                valid_metric_header = ["# Epoch", "AUROC", "F1_weighted", "Val_loss"]
                test_metric_header = ["# Best Epoch", "AUROC", "F1_weighted",
                                      "Accuracy", "Test_loss"]
        elif self.use_multitask:
            valid_metric_header = ["# Epoch", "AUROC", "AUPRC", "Val_loss", "R²"]
            test_metric_header = ["# Best Epoch", "AUROC", "AUPRC", "F1", "Sensitivity", "Specificity", "Accuracy",
                                  "Threshold", "Test_loss", "R²"]
        else:
            valid_metric_header = ["# Epoch", "AUROC", "AUPRC", "Val_loss"]
            test_metric_header = ["# Best Epoch", "AUROC", "AUPRC", "F1", "Sensitivity", "Specificity", "Accuracy",
                                  "Threshold", "Test_loss"]

        if not self.is_da:
            if self.use_multitask:
                train_metric_header = ["# Epoch", "Train_loss", "Class_loss", "Reg_loss"]
            else:
                train_metric_header = ["# Epoch", "Train_loss"]
        else:
            train_metric_header = ["# Epoch", "Train_loss", "Model_loss", "epoch_lamb_da", "da_loss"]
        self.val_table = PrettyTable(valid_metric_header)
        self.test_table = PrettyTable(test_metric_header)
        self.train_table = PrettyTable(train_metric_header)

        self.original_random = config["DA"]["ORIGINAL_RANDOM"]

    def da_lambda_decay(self):
        delta_epoch = self.current_epoch - self.da_init_epoch
        non_init_epoch = self.epochs - self.da_init_epoch
        p = (self.current_epoch + delta_epoch * self.nb_training) / (
                non_init_epoch * self.nb_training
        )
        grow_fact = 2.0 / (1.0 + np.exp(-10 * p)) - 1
        return self.init_lamb_da * grow_fact

    def train(self):
        float2str = lambda x: '%0.4f' % x
        for i in range(self.epochs):
            self.current_epoch += 1
            if not self.is_da:
                epoch_output = self.train_epoch()
                if self.use_multitask:
                    train_loss, class_loss, reg_loss = epoch_output
                    train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss, class_loss, reg_loss]))
                    if self.experiment:
                        self.experiment.log_metric("train_epoch total loss", train_loss, epoch=self.current_epoch)
                        self.experiment.log_metric("train_epoch class loss", class_loss, epoch=self.current_epoch)
                        self.experiment.log_metric("train_epoch reg loss", reg_loss, epoch=self.current_epoch)
                else:
                    train_loss = epoch_output
                    train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss]))
                    if self.experiment:
                        self.experiment.log_metric("train_epoch model loss", train_loss, epoch=self.current_epoch)
            else:
                train_loss, model_loss, da_loss, epoch_lamb = self.train_da_epoch()
                train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss, model_loss,
                                                                                        epoch_lamb, da_loss]))
                self.train_model_loss_epoch.append(model_loss)
                self.train_da_loss_epoch.append(da_loss)
                if self.experiment:
                    self.experiment.log_metric("train_epoch total loss", train_loss, epoch=self.current_epoch)
                    self.experiment.log_metric("train_epoch model loss", model_loss, epoch=self.current_epoch)
                    if self.current_epoch >= self.da_init_epoch:
                        self.experiment.log_metric("train_epoch da loss", da_loss, epoch=self.current_epoch)
            self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)

            # Validation
            val_output = self.test(dataloader="val")
            if self.use_multitask:
                auroc, val_metric2, val_loss, val_r2 = val_output
            else:
                auroc, val_metric2, val_loss = val_output
                val_r2 = None
            # val_metric2 is AUPRC for binary, F1_weighted for 3-class

            if self.experiment:
                self.experiment.log_metric("valid_epoch model loss", val_loss, epoch=self.current_epoch)
                self.experiment.log_metric("valid_epoch auroc", auroc, epoch=self.current_epoch)
                if self.use_multitask and val_r2 is not None:
                    self.experiment.log_metric("valid_epoch r2", val_r2, epoch=self.current_epoch)

            # Add validation results to table
            if self.use_multitask:
                val_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [auroc, val_metric2, val_loss, val_r2]))
            else:
                val_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [auroc, val_metric2, val_loss]))
            self.val_table.add_row(val_lst)
            self.val_loss_epoch.append(val_loss)
            self.val_auroc_epoch.append(auroc)

            # Choose best model based on validation loss (better for multitask)
            if self.use_multitask:
                # For multitask: use validation loss (combines classification + regression)
                if val_loss <= self.best_val_loss:
                    self.best_model = copy.deepcopy(self.model)
                    self.best_val_loss = val_loss
                    self.best_auroc = auroc
                    self.best_epoch = self.current_epoch
                    r2_str = f', R²: {val_r2:.4f}' if val_r2 is not None else ''
                    print(f"  New best model! Val Loss: {val_loss:.4f}, AUROC: {auroc:.4f}{r2_str}")
            else:
                # For single-task: use AUROC
                if auroc >= self.best_auroc:
                    self.best_model = copy.deepcopy(self.model)
                    self.best_auroc = auroc
                    self.best_epoch = self.current_epoch
                    print(f"  New best model! AUROC: {auroc:.4f}")

            # Print validation results
            metric2_name = "F1_weighted" if self.n_class >= 3 else "AUPRC"
            val_msg = f'Validation at Epoch {self.current_epoch} with validation loss {val_loss:.4f}, AUROC {auroc:.4f}, {metric2_name} {val_metric2:.4f}'
            if self.use_multitask and val_r2 is not None:
                val_msg += f', R² {val_r2:.4f}'
            print(val_msg)

        # Test on best model
        test_output = self.test(dataloader="test")

        if self.n_class >= 3:
            # Multi-class test output (3, 10, or any n_class)
            if self.use_multitask:
                auroc, f1_w, accuracy, test_loss, per_class_recall, test_r2 = test_output
                test_lst = ["epoch " + str(self.best_epoch)] + list(map(float2str,
                    [auroc, f1_w, accuracy, test_loss, test_r2]))
            else:
                auroc, f1_w, accuracy, test_loss, per_class_recall = test_output
                test_r2 = None
                test_lst = ["epoch " + str(self.best_epoch)] + list(map(float2str,
                    [auroc, f1_w, accuracy, test_loss]))
            self.test_table.add_row(test_lst)

            recall_str = ', '.join([f'c{i}={r:.4f}' for i, r in enumerate(per_class_recall)])
            test_msg = (f'Test at Best Model of Epoch {self.best_epoch} with test loss {test_loss:.4f}, '
                        f'AUROC {auroc:.4f}, F1_weighted {f1_w:.4f}, Accuracy {accuracy:.4f}, '
                        f'Per-class Recall: [{recall_str}]')
            if self.use_multitask and test_r2 is not None:
                test_msg += f', R² {test_r2:.4f}'
            print(test_msg)

            self.test_metrics["auroc"] = auroc
            self.test_metrics["f1_weighted"] = f1_w
            self.test_metrics["accuracy"] = accuracy
            self.test_metrics["test_loss"] = test_loss
            self.test_metrics["best_epoch"] = self.best_epoch
            for i, r in enumerate(per_class_recall):
                self.test_metrics[f"recall_c{i}"] = r
            if self.use_multitask and test_r2 is not None:
                self.test_metrics["R2"] = test_r2
        else:
            # Binary test output (original)
            if self.use_multitask:
                auroc, auprc, f1, sensitivity, specificity, accuracy, test_loss, thred_optim, precision, test_r2 = test_output
            else:
                auroc, auprc, f1, sensitivity, specificity, accuracy, test_loss, thred_optim, precision = test_output
                test_r2 = None
            # Add test results to table
            if self.use_multitask:
                test_lst = ["epoch " + str(self.best_epoch)] + list(map(float2str, [auroc, auprc, f1, sensitivity, specificity,
                                                                                    accuracy, thred_optim, test_loss, test_r2]))
            else:
                test_lst = ["epoch " + str(self.best_epoch)] + list(map(float2str, [auroc, auprc, f1, sensitivity, specificity,
                                                                                    accuracy, thred_optim, test_loss]))
            self.test_table.add_row(test_lst)

            test_msg = f'Test at Best Model of Epoch {self.best_epoch} with test loss {test_loss:.4f}, AUROC {auroc:.4f}, AUPRC {auprc:.4f}, Sensitivity {sensitivity:.4f}, Specificity {specificity:.4f}, Accuracy {accuracy:.4f}, Thred_optim {thred_optim:.4f}'
            if self.use_multitask and test_r2 is not None:
                test_msg += f', R² {test_r2:.4f}'
            print(test_msg)

            self.test_metrics["auroc"] = auroc
            self.test_metrics["auprc"] = auprc
            self.test_metrics["test_loss"] = test_loss
            self.test_metrics["sensitivity"] = sensitivity
            self.test_metrics["specificity"] = specificity
            self.test_metrics["accuracy"] = accuracy
            self.test_metrics["thred_optim"] = thred_optim
            self.test_metrics["best_epoch"] = self.best_epoch
            self.test_metrics["F1"] = f1
            self.test_metrics["Precision"] = precision
            if self.use_multitask and test_r2 is not None:
                self.test_metrics["R2"] = test_r2

        self.save_result()
        if self.experiment:
            self.experiment.log_metric("valid_best_auroc", self.best_auroc)
            self.experiment.log_metric("valid_best_epoch", self.best_epoch)
            self.experiment.log_metric("test_auroc", self.test_metrics["auroc"])
            self.experiment.log_metric("test_accuracy", self.test_metrics["accuracy"])
            if self.n_class >= 3:
                self.experiment.log_metric("test_f1_weighted", self.test_metrics["f1_weighted"])
            else:
                self.experiment.log_metric("test_auprc", self.test_metrics["auprc"])
                self.experiment.log_metric("test_sensitivity", self.test_metrics["sensitivity"])
                self.experiment.log_metric("test_specificity", self.test_metrics["specificity"])
                self.experiment.log_metric("test_threshold", self.test_metrics["thred_optim"])
                self.experiment.log_metric("test_f1", self.test_metrics["F1"])
                self.experiment.log_metric("test_precision", self.test_metrics["Precision"])
            if self.use_multitask and "R2" in self.test_metrics:
                self.experiment.log_metric("test_r2", self.test_metrics["R2"])
        return self.test_metrics

    def save_result(self):
        if self.config["RESULT"]["SAVE_MODEL"]:
            torch.save(self.best_model.state_dict(),
                       os.path.join(self.output_dir, f"best_model_epoch_{self.best_epoch}.pth"))
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"model_epoch_{self.current_epoch}.pth"))
        state = {
            "train_epoch_loss": self.train_loss_epoch,
            "val_epoch_loss": self.val_loss_epoch,
            "test_metrics": self.test_metrics,
            "config": self.config
        }
        if self.is_da:
            state["train_model_loss"] = self.train_model_loss_epoch
            state["train_da_loss"] = self.train_da_loss_epoch
            state["da_init_epoch"] = self.da_init_epoch
        torch.save(state, os.path.join(self.output_dir, f"result_metrics.pt"))

        val_prettytable_file = os.path.join(self.output_dir, "valid_markdowntable.txt")
        test_prettytable_file = os.path.join(self.output_dir, "test_markdowntable.txt")
        train_prettytable_file = os.path.join(self.output_dir, "train_markdowntable.txt")
        with open(val_prettytable_file, 'w') as fp:
            fp.write(self.val_table.get_string())
        with open(test_prettytable_file, 'w') as fp:
            fp.write(self.test_table.get_string())
        with open(train_prettytable_file, "w") as fp:
            fp.write(self.train_table.get_string())

    def _compute_entropy_weights(self, logits):
        entropy = entropy_logits(logits)
        entropy = ReverseLayerF.apply(entropy, self.alpha)
        entropy_w = 1.0 + torch.exp(-entropy)
        return entropy_w

    def _compute_mmd(self, source_features, target_features, kernel_mul=2.0, kernel_num=5):
        """
        Compute Maximum Mean Discrepancy (MMD) between source and target domains.

        Args:
            source_features: Source domain features (batch_size_src, feature_dim)
            target_features: Target domain features (batch_size_tgt, feature_dim)
            kernel_mul: Kernel bandwidth multiplier
            kernel_num: Number of kernels to use

        Returns:
            mmd_loss: MMD distance between source and target distributions
        """
        batch_size_src = source_features.size(0)
        batch_size_tgt = target_features.size(0)

        # Concatenate source and target features
        features = torch.cat([source_features, target_features], dim=0)

        # Compute pairwise distances
        n = features.size(0)
        features_expanded_1 = features.unsqueeze(1).expand(n, n, -1)
        features_expanded_2 = features.unsqueeze(0).expand(n, n, -1)

        # L2 distance
        pairwise_distances = torch.sum((features_expanded_1 - features_expanded_2) ** 2, dim=2)

        # Compute bandwidth for Gaussian kernel
        # Use median heuristic
        bandwidth = torch.median(pairwise_distances[pairwise_distances > 0])
        bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

        # Compute multi-kernel MMD
        kernel_val = torch.zeros_like(pairwise_distances)
        for bandwidth in bandwidth_list:
            kernel_val += torch.exp(-pairwise_distances / (2 * bandwidth))

        # Split kernel matrix into 4 parts
        # XX: source-source, XY: source-target, YX: target-source, YY: target-target
        XX = kernel_val[:batch_size_src, :batch_size_src]
        XY = kernel_val[:batch_size_src, batch_size_src:]
        YX = kernel_val[batch_size_src:, :batch_size_src]
        YY = kernel_val[batch_size_src:, batch_size_src:]

        # MMD^2 = E[k(x,x')] - 2*E[k(x,y)] + E[k(y,y')]
        mmd_loss = torch.mean(XX) - 2 * torch.mean(XY) + torch.mean(YY)

        return mmd_loss

    def _clone_model_params(self, model):
        """
        Clone model parameters for MAML inner loop adaptation.

        Args:
            model: The model whose parameters to clone

        Returns:
            Dictionary of cloned parameters with gradient tracking enabled
        """
        return {
            name: param.clone().requires_grad_(True)
            for name, param in model.named_parameters()
        }

    def _forward_with_params(self, model, v_d, v_p, params):
        """
        Perform forward pass using custom parameters (for MAML inner loop).

        Args:
            model: The model to use
            v_d: Drug data (will be cloned if it's a DGL graph to avoid modification)
            v_p: Protein data
            params: Dictionary of parameters to use instead of model.parameters()

        Returns:
            Model output (v_d, v_p, f, score, score_for_da)
        """
        # Clone DGL graph if needed, because model.forward() uses .pop('h') which modifies the graph
        # In MAML's inner loop, we call forward K times with the same graph, so we need fresh copies
        if hasattr(v_d, 'ndata') and hasattr(v_d, 'clone'):
            # DGL graph - clone it to avoid modification
            v_d = v_d.clone()

        # Temporarily replace model parameters
        original_params = {}
        for name, param in model.named_parameters():
            original_params[name] = param.data.clone()
            param.data = params[name].data

        # Forward pass
        output = model(v_d, v_p, mode="train")

        # Restore original parameters
        for name, param in model.named_parameters():
            param.data = original_params[name]

        return output

    def _extract_task_data(self, batch_data, task_idx, task_size):
        """
        Extract data for a single task from a batched MAML input.

        Args:
            batch_data: Dictionary containing 'v_d', 'v_p', 'labels'
                       For MAML, v_d is kept as a list (not batched yet) to avoid DGL issues
            task_idx: Index of the task to extract
            task_size: Number of samples per task

        Returns:
            Tuple of (v_d, v_p, labels) for the specified task
        """
        start_idx = task_idx * task_size
        end_idx = start_idx + task_size

        # Handle drug data
        v_d = batch_data['v_d']
        if isinstance(v_d, list):
            # Graph mode: v_d is a list of DGL graphs - batch only the subset we need
            import dgl
            graphs_subset = v_d[start_idx:end_idx]
            # Batch the graphs
            # IMPORTANT: We need to create fresh copies because the model's forward pass
            # uses .pop('h') which modifies the graph, and in MAML we call forward K times
            # with the same graphs in the inner loop
            v_d_task = dgl.batch(graphs_subset)
        elif isinstance(v_d, tuple):
            # SELFIES mode: v_d is tuple of tensors like (drug_idx, drug_len) or (drug_idx, drug_feat, drug_len)
            # This case shouldn't happen with MAML since we keep everything as lists, but handle it for safety
            v_d_task = tuple(item[start_idx:end_idx] for item in v_d)
        else:
            # Shouldn't reach here, but fallback to direct slicing
            v_d_task = v_d[start_idx:end_idx]

        # Handle protein data (can be tuple for BiLSTM or single tensor for CNN)
        v_p = batch_data['v_p']
        if isinstance(v_p, tuple):
            # BiLSTM mode: v_p is tuple (p_idx, p_feat, p_len)
            v_p_task = tuple(item[start_idx:end_idx] for item in v_p)
        else:
            # CNN mode: v_p is single tensor
            v_p_task = v_p[start_idx:end_idx]

        labels = batch_data['labels'][start_idx:end_idx]

        return (
            self._to_device(v_d_task),
            self._to_device(v_p_task),
            labels.float().to(self.device)
        )

    def _prepare_target_data(self, target_samples):
        """
        Prepare target domain samples for MMD computation.

        Args:
            target_samples: List of (v_d, v_p, y) tuples from target domain

        Returns:
            Tuple of (v_d, v_p, labels) moved to device
        """
        # Import collate function
        # collate_selfies_fn handles both graph and SELFIES modes automatically
        from dataloader import collate_selfies_fn

        # Collate target samples (works for both graph and SELFIES modes)
        v_d, v_p, labels = collate_selfies_fn(target_samples)

        return (
            self._to_device(v_d),
            self._to_device(v_p),
            labels.float().to(self.device)
        )

    def _maml_meta_step(self, batch_data):
        """
        Execute one MAML meta-learning step with MMD domain adaptation.

        Args:
            batch_data: Dictionary containing:
                {
                    'source_task': {
                        'protein_ids': list,
                        'support': {'v_d', 'v_p', 'labels'},
                        'query': {'v_d', 'v_p', 'labels'}
                    },
                    'target_samples': list of (v_d, v_p, y)
                }

        Returns:
            meta_loss: Combined task loss + MMD loss

        Algorithm:
            1. For each task:
               a. Clone parameters θ' ← θ
               b. Inner loop: Update θ' on support set (K steps)
               c. Evaluate on query set using θ'
            2. Compute MMD between source and target features
            3. Meta-loss = avg(query losses) + λ * MMD
        """
        maml_cfg = self.config["MAML"]
        inner_lr = maml_cfg["INNER_LR"]
        inner_steps = maml_cfg["INNER_STEPS"]
        first_order = maml_cfg["FIRST_ORDER"]
        mmd_weight = maml_cfg["MMD_WEIGHT"]
        support_size = maml_cfg["SUPPORT_SIZE"]
        query_size = maml_cfg["QUERY_SIZE"]

        # Extract source tasks and target samples
        source_task = batch_data['source_task']
        target_samples = batch_data['target_samples']

        num_tasks = len(source_task['protein_ids'])

        meta_task_losses = []
        source_features_all = []

        # === Inner Loop: Adapt to each task ===
        for task_idx in range(num_tasks):
            # Extract support and query data for this task
            support_v_d, support_v_p, support_labels = self._extract_task_data(
                source_task['support'], task_idx, support_size
            )
            query_v_d, query_v_p, query_labels = self._extract_task_data(
                source_task['query'], task_idx, query_size
            )

            # Clone model parameters for inner loop
            fast_weights = self._clone_model_params(self.model)

            # Inner loop: K steps of gradient descent on support set
            for step in range(inner_steps):
                # Forward pass with current fast_weights
                _, _, _, score_sup, _ = self._forward_with_params(
                    self.model, support_v_d, support_v_p, fast_weights
                )

                # Compute support loss
                if self.n_class == 1 or score_sup.size(1) == 1:
                    _, sup_loss = binary_cross_entropy(score_sup, support_labels)
                else:
                    _, sup_loss = cross_entropy_logits(score_sup, support_labels, None)

                # Compute gradients w.r.t. fast_weights
                grads = torch.autograd.grad(
                    sup_loss,
                    fast_weights.values(),
                    create_graph=not first_order,  # Second-order for MAML, first-order for efficiency
                    retain_graph=True if step < inner_steps - 1 else False,
                    allow_unused=True  # Some parameters may not be used in forward pass
                )

                # Update fast_weights (inner loop SGD)
                # Skip parameters with None gradients (unused in forward pass)
                fast_weights = {
                    name: param - inner_lr * grad if grad is not None else param
                    for (name, param), grad in zip(fast_weights.items(), grads)
                }

            # Evaluate on query set using adapted parameters
            _, _, f_qry, score_qry, _ = self._forward_with_params(
                self.model, query_v_d, query_v_p, fast_weights
            )

            # Compute query loss (meta-objective)
            if self.n_class == 1 or score_qry.size(1) == 1:
                _, qry_loss = binary_cross_entropy(score_qry, query_labels)
            else:
                _, qry_loss = cross_entropy_logits(score_qry, query_labels, None)

            meta_task_losses.append(qry_loss)
            source_features_all.append(f_qry)

        # === Compute Meta Task Loss ===
        meta_task_loss = torch.stack(meta_task_losses).mean()

        # === Compute MMD Loss ===
        # Forward pass on target domain to extract features
        target_v_d, target_v_p, _ = self._prepare_target_data(target_samples)
        _, _, f_tgt, _, _ = self.model(target_v_d, target_v_p, mode="train")

        # Concatenate source features from all tasks
        source_features = torch.cat(source_features_all, dim=0)

        # Compute MMD
        mmd_loss = self._compute_mmd(
            source_features,
            f_tgt,
            kernel_mul=maml_cfg["MMD_KERNEL_MUL"],
            kernel_num=maml_cfg["MMD_KERNEL_NUM"]
        )

        # === Total Meta Loss ===
        total_meta_loss = meta_task_loss + mmd_weight * mmd_loss

        # Logging
        if self.experiment:
            self.experiment.log_metric("maml_task_loss", meta_task_loss.item(), step=self.step)
            self.experiment.log_metric("maml_mmd_loss", mmd_loss.item(), step=self.step)
            self.experiment.log_metric("maml_total_loss", total_meta_loss.item(), step=self.step)

        return total_meta_loss

    def set_expert_models(self, expert_low_model, expert_high_model):
        """Set expert models for knowledge distillation."""
        self.expert_low_model = expert_low_model
        self.expert_high_model = expert_high_model
        self.expert_low_model.eval()
        self.expert_high_model.eval()
        print(f"Expert models loaded for knowledge distillation")

    def train_epoch(self):
        self.model.train()
        loss_epoch = 0
        class_loss_epoch = 0  # Classification loss
        reg_loss_epoch = 0    # Regression loss
        num_batches = len(self.train_dataloader)

        # Check if using support sets
        use_support = self.config.get("SUPPORT_SET", {}).get("USE", False)

        for i, batch_data in enumerate(tqdm(self.train_dataloader)):
            self.step += 1

            if use_support:
                # Unpack query and support data
                query_data = batch_data['query']
                support_data = batch_data['support']

                v_d, v_p, labels, z_values = query_data
                support_v_d, support_v_p, support_labels, support_z_values = support_data

                # Move query data to device
                v_d = self._to_device(v_d)
                v_p = self._to_device(v_p)
                labels = labels.float().to(self.device)
                z_values = z_values.float().to(self.device)

                # Move support data to device
                support_v_d = self._to_device(support_v_d)
                support_v_p = self._to_device(support_v_p)
                support_labels = support_labels.float().to(self.device)

                # Forward with support set
                self.optim.zero_grad()
                forward_output = self.model(
                    v_d, v_p,
                    support_bg_d=support_v_d,
                    support_v_p=support_v_p,
                    support_labels=support_labels,
                    mode="train"
                )
                if self.use_multitask:
                    v_d, v_p, f, score, score_for_da, reg_output = forward_output
                else:
                    v_d, v_p, f, score, score_for_da = forward_output
            else:
                # Original behavior (no support set)
                v_d, v_p, labels, z_values = batch_data
                v_d = self._to_device(v_d)
                v_p = self._to_device(v_p)
                labels = labels.float().to(self.device)
                z_values = z_values.float().to(self.device)

                # For distillation: get expert predictions BEFORE student forward
                # (because forward modifies v_d from DGL graph to tensor)
                expert_low_reg = None
                expert_high_reg = None
                if self.use_distillation and self.expert_low_model is not None and self.expert_high_model is not None:
                    with torch.no_grad():
                        _, _, _, _, _, expert_low_reg = self.expert_low_model(v_d, v_p, mode="train")
                        # Need to reload v_d since expert_low consumed it
                        v_d = self._to_device(batch_data[0])
                        _, _, _, _, _, expert_high_reg = self.expert_high_model(v_d, v_p, mode="train")
                        # Reload again for student model
                        v_d = self._to_device(batch_data[0])

                self.optim.zero_grad()
                forward_output = self.model(v_d, v_p, mode="train")
                if self.use_multitask:
                    v_d, v_p, f, score, score_for_da, reg_output = forward_output
                else:
                    v_d, v_p, f, score, score_for_da = forward_output

            # Compute classification loss
            if self.n_class == 1 or score.size(1) == 1:
                n, class_loss = binary_cross_entropy(score, labels)
            else:
                n, class_loss = cross_entropy_logits(score, labels)

            # Multitask: compute total loss
            if self.use_multitask:
                reg_loss_fn = torch.nn.MSELoss()

                # Knowledge Distillation: use expert predictions as soft targets
                if self.use_distillation and expert_low_reg is not None and expert_high_reg is not None:

                    # Create soft targets based on class labels
                    # Y=0 (inactive): use expert_low's prediction
                    # Y=1 (active): use expert_high's prediction
                    expert_targets = torch.where(
                        labels.unsqueeze(1) < 0.5,
                        expert_low_reg,
                        expert_high_reg
                    ).squeeze()

                    # Combine ground truth and expert predictions
                    reg_loss_gt = reg_loss_fn(reg_output.squeeze(), z_values)
                    reg_loss_distill = reg_loss_fn(reg_output.squeeze(), expert_targets)
                    reg_loss = (1 - self.distill_alpha) * reg_loss_gt + self.distill_alpha * reg_loss_distill
                else:
                    # Standard regression loss
                    reg_loss = reg_loss_fn(reg_output.squeeze(), z_values)

                # Total loss = alpha * classification loss + beta * regression loss
                loss = self.multitask_alpha * class_loss + self.multitask_beta * reg_loss

                class_loss_epoch += class_loss.item()
                reg_loss_epoch += reg_loss.item()
            else:
                loss = class_loss

            loss.backward()
            self.optim.step()
            loss_epoch += loss.item()

            if self.experiment:
                self.experiment.log_metric("train_step total loss", loss.item(), step=self.step)
                if self.use_multitask:
                    self.experiment.log_metric("train_step class loss", class_loss.item(), step=self.step)
                    self.experiment.log_metric("train_step reg loss", reg_loss.item(), step=self.step)

        loss_epoch = loss_epoch / num_batches

        if self.use_multitask:
            class_loss_epoch = class_loss_epoch / num_batches
            reg_loss_epoch = reg_loss_epoch / num_batches
            print(f'Training at Epoch {self.current_epoch} - Total: {loss_epoch:.4f}, Class: {class_loss_epoch:.4f}, Reg: {reg_loss_epoch:.4f}')
            return loss_epoch, class_loss_epoch, reg_loss_epoch
        else:
            print('Training at Epoch ' + str(self.current_epoch) + ' with training loss ' + str(loss_epoch))
            return loss_epoch

    def train_da_epoch(self):
        self.model.train()
        total_loss_epoch = 0
        model_loss_epoch = 0
        da_loss_epoch = 0
        epoch_lamb_da = 0
        if self.current_epoch >= self.da_init_epoch:
            # epoch_lamb_da = self.da_lambda_decay()
            epoch_lamb_da = 1
            if self.experiment:
                self.experiment.log_metric("DA loss lambda", epoch_lamb_da, epoch=self.current_epoch)
        num_batches = len(self.train_dataloader)

        # Check if using MAML meta-learning
        use_maml = (self.da_method == "MetaLearning" and
                    self.config.get("MAML", {}).get("ENABLE", False))

        if use_maml:
            # MAML uses a different training loop
            for i, batch_data in enumerate(tqdm(self.train_dataloader)):
                self.step += 1
                self.optim.zero_grad()

                # MAML meta-step (includes inner loop + MMD)
                meta_loss = self._maml_meta_step(batch_data)

                # Backpropagation and optimization
                meta_loss.backward()
                self.optim.step()

                total_loss_epoch += meta_loss.item()
                model_loss_epoch += meta_loss.item()  # For MAML, all loss is "model loss"
                if self.current_epoch >= self.da_init_epoch:
                    da_loss_epoch += 0  # MAML loss is already combined

            total_loss_epoch = total_loss_epoch / num_batches
            model_loss_epoch = model_loss_epoch / num_batches
            da_loss_epoch = 0  # MAML doesn't separate DA loss

            print(f'Training at Epoch {self.current_epoch} with MAML meta loss {total_loss_epoch:.4f}')
            return total_loss_epoch, model_loss_epoch, da_loss_epoch, epoch_lamb_da

        # Standard DA training (CDAN, DANN, MMD)
        # Check if using support sets
        use_support = self.config.get("SUPPORT_SET", {}).get("USE", False)

        for i, (batch_s, batch_t) in enumerate(tqdm(self.train_dataloader)):
            self.step += 1

            # Unpack source batch
            if use_support:
                # With support sets, batch is a dict with 'query' and 'support' keys
                query_data_s = batch_s['query']
                if self.use_multitask:
                    v_d, v_p, labels, z_values = query_data_s
                else:
                    v_d, v_p, labels = query_data_s
            else:
                # Without support sets, batch is a tuple
                if self.use_multitask:
                    v_d, v_p, labels, z_values = batch_s
                else:
                    v_d, v_p, labels = batch_s

            # Move source data to device
            if isinstance(v_p, tuple):
                v_p = tuple(item.to(self.device) for item in v_p)
            else:
                v_p = self._to_device(v_p)
            v_d = self._to_device(v_d)
            labels = labels.float().to(self.device)
            if self.use_multitask:
                z_values = z_values.float().to(self.device)

            # Unpack target batch
            if use_support:
                query_data_t = batch_t['query']
                if self.use_multitask:
                    v_d_t, v_p_t, labels_t, z_values_t = query_data_t
                else:
                    v_d_t, v_p_t, labels_t = query_data_t
            else:
                if self.use_multitask:
                    v_d_t, v_p_t, labels_t, z_values_t = batch_t
                else:
                    v_d_t, v_p_t, labels_t = batch_t

            # Move target data to device
            v_d_t = self._to_device(v_d_t)
            if isinstance(v_p_t, tuple):
                v_p_t = tuple(item.to(self.device) for item in v_p_t)
            else:
                v_p_t = self._to_device(v_p_t)

            self.optim.zero_grad()
            if self.optim_da is not None:
                self.optim_da.zero_grad()

            # Forward pass for source domain
            if use_support:
                support_data_s = batch_s['support']
                support_v_d, support_v_p, support_labels = support_data_s
                # Move support data to device
                support_v_d = self._to_device(support_v_d)
                if isinstance(support_v_p, tuple):
                    support_v_p = tuple(item.to(self.device) for item in support_v_p)
                else:
                    support_v_p = self._to_device(support_v_p)
                support_labels = support_labels.float().to(self.device)
                v_d, v_p, f, score, score_for_da = self.model(
                    v_d, v_p,
                    support_bg_d=support_v_d,
                    support_v_p=support_v_p,
                    support_labels=support_labels,
                    mode="train"
                )
            else:
                forward_output = self.model(v_d, v_p, mode="train")
                if self.use_multitask:
                    v_d, v_p, f, score, score_for_da, reg_output = forward_output
                else:
                    v_d, v_p, f, score, score_for_da = forward_output
            # Check both n_class and actual score shape (support set may output [B,1])
            if self.n_class == 1 or score.size(1) == 1:
                n, class_loss = binary_cross_entropy(score, labels)
            else:
                n, class_loss = cross_entropy_logits(score, labels)

            # Compute model loss (classification + optional regression)
            if self.use_multitask:
                reg_loss_fn = torch.nn.MSELoss()
                reg_loss = reg_loss_fn(reg_output.squeeze(), z_values)
                model_loss = self.multitask_alpha * class_loss + self.multitask_beta * reg_loss
            else:
                model_loss = class_loss
            if self.current_epoch >= self.da_init_epoch:
                # Forward pass for target domain
                if use_support:
                    support_data_t = batch_t['support']
                    support_v_d_t, support_v_p_t, support_labels_t = support_data_t
                    # Move support data to device
                    support_v_d_t = self._to_device(support_v_d_t)
                    if isinstance(support_v_p_t, tuple):
                        support_v_p_t = tuple(item.to(self.device) for item in support_v_p_t)
                    else:
                        support_v_p_t = self._to_device(support_v_p_t)
                    support_labels_t = support_labels_t.float().to(self.device)
                    v_d_t, v_p_t, f_t, t_score, t_score_for_da = self.model(
                        v_d_t, v_p_t,
                        support_bg_d=support_v_d_t,
                        support_v_p=support_v_p_t,
                        support_labels=support_labels_t,
                        mode="train"
                    )
                else:
                    forward_output_t = self.model(v_d_t, v_p_t, mode="train")
                    if self.use_multitask:
                        v_d_t, v_p_t, f_t, t_score, t_score_for_da, _ = forward_output_t
                    else:
                        v_d_t, v_p_t, f_t, t_score, t_score_for_da = forward_output_t
                if self.da_method == "CDAN":
                    reverse_f = ReverseLayerF.apply(f, self.alpha)
                    # Use score_for_da (full 2-class) for CDAN, not ensemble score
                    softmax_output = torch.nn.Softmax(dim=1)(score_for_da)
                    softmax_output = softmax_output.detach()
                    # reverse_output = ReverseLayerF.apply(softmax_output, self.alpha)
                    if self.original_random:
                        random_out = self.random_layer.forward([reverse_f, softmax_output])
                        adv_output_src_score = self.domain_dmm(random_out.view(-1, random_out.size(1)))
                    else:
                        feature = torch.bmm(softmax_output.unsqueeze(2), reverse_f.unsqueeze(1))
                        feature = feature.view(-1, softmax_output.size(1) * reverse_f.size(1))
                        if self.random_layer:
                            random_out = self.random_layer.forward(feature)
                            adv_output_src_score = self.domain_dmm(random_out)
                        else:
                            adv_output_src_score = self.domain_dmm(feature)

                    reverse_f_t = ReverseLayerF.apply(f_t, self.alpha)
                    # Use t_score_for_da (full 2-class) for CDAN, not ensemble score
                    softmax_output_t = torch.nn.Softmax(dim=1)(t_score_for_da)
                    softmax_output_t = softmax_output_t.detach()
                    # reverse_output_t = ReverseLayerF.apply(softmax_output_t, self.alpha)
                    if self.original_random:
                        random_out_t = self.random_layer.forward([reverse_f_t, softmax_output_t])
                        adv_output_tgt_score = self.domain_dmm(random_out_t.view(-1, random_out_t.size(1)))
                    else:
                        feature_t = torch.bmm(softmax_output_t.unsqueeze(2), reverse_f_t.unsqueeze(1))
                        feature_t = feature_t.view(-1, softmax_output_t.size(1) * reverse_f_t.size(1))
                        if self.random_layer:
                            random_out_t = self.random_layer.forward(feature_t)
                            adv_output_tgt_score = self.domain_dmm(random_out_t)
                        else:
                            adv_output_tgt_score = self.domain_dmm(feature_t)

                    if self.use_da_entropy:
                        entropy_src = self._compute_entropy_weights(score)
                        entropy_tgt = self._compute_entropy_weights(t_score)
                        src_weight = entropy_src / torch.sum(entropy_src)
                        tgt_weight = entropy_tgt / torch.sum(entropy_tgt)
                    else:
                        src_weight = None
                        tgt_weight = None

                    # Use actual batch size from each domain's data (source and target may differ)
                    actual_batch_size_src = adv_output_src_score.size(0)
                    actual_batch_size_tgt = adv_output_tgt_score.size(0)
                    n_src, loss_cdan_src = cross_entropy_logits(adv_output_src_score, torch.zeros(actual_batch_size_src).to(self.device),
                                                                src_weight)
                    n_tgt, loss_cdan_tgt = cross_entropy_logits(adv_output_tgt_score, torch.ones(actual_batch_size_tgt).to(self.device),
                                                                tgt_weight)
                    da_loss = loss_cdan_src + loss_cdan_tgt

                elif self.da_method == "DANN":
                    # DANN: Domain-Adversarial Neural Network
                    # Use gradient reversal layer to learn domain-invariant features
                    reverse_f = ReverseLayerF.apply(f, self.alpha)
                    reverse_f_t = ReverseLayerF.apply(f_t, self.alpha)

                    # Domain classifier prediction
                    domain_pred_src = self.domain_dmm(reverse_f)
                    domain_pred_tgt = self.domain_dmm(reverse_f_t)

                    # Actual batch sizes
                    actual_batch_size_src = domain_pred_src.size(0)
                    actual_batch_size_tgt = domain_pred_tgt.size(0)

                    # Domain labels: source=0, target=1
                    n_src, loss_dann_src = cross_entropy_logits(
                        domain_pred_src,
                        torch.zeros(actual_batch_size_src).to(self.device),
                        None
                    )
                    n_tgt, loss_dann_tgt = cross_entropy_logits(
                        domain_pred_tgt,
                        torch.ones(actual_batch_size_tgt).to(self.device),
                        None
                    )
                    da_loss = loss_dann_src + loss_dann_tgt

                elif self.da_method == "MMD":
                    # MMD: Maximum Mean Discrepancy
                    # Minimize distribution distance between source and target domains
                    da_loss = self._compute_mmd(f, f_t)

                elif self.da_method == "MetaLearning":
                    # This branch should not be reached when MAML is enabled
                    # MAML training is handled at the beginning of train_da_epoch()
                    if not self.config.get("MAML", {}).get("ENABLE", False):
                        raise ValueError("MetaLearning method requires MAML.ENABLE=True in config")
                    else:
                        raise RuntimeError("MAML training should be handled in the special loop above. "
                                         "This error suggests a logic issue in train_da_epoch().")

                else:
                    raise ValueError(f"The da method {self.da_method} is not supported. "
                                   f"Supported methods: CDAN, DANN, MMD")
                loss = model_loss + da_loss
            else:
                loss = model_loss
            loss.backward()
            self.optim.step()
            if self.optim_da is not None:
                self.optim_da.step()
            total_loss_epoch += loss.item()
            model_loss_epoch += model_loss.item()
            if self.experiment:
                self.experiment.log_metric("train_step model loss", model_loss.item(), step=self.step)
                self.experiment.log_metric("train_step total loss", loss.item(), step=self.step)
            if self.current_epoch >= self.da_init_epoch:
                da_loss_epoch += da_loss.item()
                if self.experiment:
                    self.experiment.log_metric("train_step da loss", da_loss.item(), step=self.step)
        total_loss_epoch = total_loss_epoch / num_batches
        model_loss_epoch = model_loss_epoch / num_batches
        da_loss_epoch = da_loss_epoch / num_batches
        if self.current_epoch < self.da_init_epoch:
            print('Training at Epoch ' + str(self.current_epoch) + ' with model training loss ' + str(total_loss_epoch))
        else:
            print('Training at Epoch ' + str(self.current_epoch) + ' model training loss ' + str(model_loss_epoch)
                  + ", da loss " + str(da_loss_epoch) + ", total training loss " + str(total_loss_epoch) + ", DA lambda " +
                  str(epoch_lamb_da))
        return total_loss_epoch, model_loss_epoch, da_loss_epoch, epoch_lamb_da

    def test(self, dataloader="test"):
        test_loss = 0
        y_label, y_pred = [], []
        z_label, z_pred = [], []  # for regression evaluation
        if dataloader == "test":
            data_loader = self.test_dataloader
        elif dataloader == "val":
            data_loader = self.val_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")
        num_batches = len(data_loader)

        # Check if using support sets
        use_support = self.config.get("SUPPORT_SET", {}).get("USE", False)

        with torch.no_grad():
            self.model.eval()
            for i, batch_data in enumerate(data_loader):
                if use_support:
                    # Unpack query and support data
                    query_data = batch_data['query']
                    support_data = batch_data['support']

                    v_d, v_p, labels, z_values = query_data
                    support_v_d, support_v_p, support_labels, support_z_values = support_data

                    # Move query data to device
                    if isinstance(v_p, tuple):
                        v_p = tuple(item.to(self.device) for item in v_p)
                    else:
                        v_p = self._to_device(v_p)
                    v_d, labels = self._to_device(v_d), labels.float().to(self.device)
                    z_values = z_values.float().to(self.device)

                    # Move support data to device
                    if isinstance(support_v_p, tuple):
                        support_v_p = tuple(item.to(self.device) for item in support_v_p)
                    else:
                        support_v_p = self._to_device(support_v_p)
                    support_v_d = self._to_device(support_v_d)
                    support_labels = support_labels.float().to(self.device)

                    # Forward with support set
                    if dataloader == "val":
                        forward_output = self.model(
                            v_d, v_p,
                            support_bg_d=support_v_d,
                            support_v_p=support_v_p,
                            support_labels=support_labels,
                            mode="train"
                        )
                        if self.use_multitask:
                            v_d, v_p, f, score, _, reg_output = forward_output
                        else:
                            v_d, v_p, f, score, _ = forward_output
                            reg_output = None
                    elif dataloader == "test":
                        forward_output = self.best_model(
                            v_d, v_p,
                            support_bg_d=support_v_d,
                            support_v_p=support_v_p,
                            support_labels=support_labels,
                            mode="eval"
                        )
                        if self.use_multitask:
                            v_d, v_p, score, att, reg_output = forward_output
                        else:
                            v_d, v_p, score, att = forward_output
                            reg_output = None
                else:
                    # Original behavior (no support set)
                    v_d, v_p, labels, z_values = batch_data
                    if isinstance(v_p, tuple):
                        v_p = tuple(item.to(self.device) for item in v_p)
                    else:
                        v_p = self._to_device(v_p)
                    v_d, labels = self._to_device(v_d), labels.float().to(self.device)
                    z_values = z_values.float().to(self.device)

                    if dataloader == "val":
                        forward_output = self.model(v_d, v_p, mode="train")
                        if self.use_multitask:
                            v_d, v_p, f, score, _, reg_output = forward_output
                        else:
                            v_d, v_p, f, score, _ = forward_output
                            reg_output = None
                    elif dataloader == "test":
                        forward_output = self.best_model(v_d, v_p, mode="train")
                        if self.use_multitask:
                            v_d, v_p, f, score, _, reg_output = forward_output
                        else:
                            v_d, v_p, f, score, _ = forward_output
                            reg_output = None

                # Compute loss (same for both modes)
                # Check both n_class and actual score shape (support set may output [B,1])
                if self.n_class == 1 or score.size(1) == 1:
                    n, loss = binary_cross_entropy(score, labels)
                else:
                    n, loss = cross_entropy_logits(score, labels)
                test_loss += loss.item()
                y_label = y_label + labels.to("cpu").tolist()
                y_pred = y_pred + n.to("cpu").tolist()

                # collect regression predictions (if multi-task learning is enabled)
                if self.use_multitask and reg_output is not None:
                    z_pred.extend(reg_output.squeeze().to("cpu").tolist())
                    z_label.extend(z_values.to("cpu").tolist())  # ground-truth CNNscore (Z values)

        test_loss = test_loss / num_batches

        # Compute R² (if multi-task learning is enabled)
        r2_score = None
        if self.use_multitask and len(z_pred) > 0:
            z_pred_array = np.array(z_pred)
            z_label_array = np.array(z_label)
            # R² = 1 - (SS_res / SS_tot)
            ss_res = np.sum((z_label_array - z_pred_array) ** 2)
            ss_tot = np.sum((z_label_array - np.mean(z_label_array)) ** 2)
            r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # ===== Multi-class metrics (3, 10, or any n_class >= 3) =====
        if self.n_class >= 3:
            y_pred_array = np.array(y_pred)   # [N, num_classes]
            y_label_array = np.array(y_label, dtype=int)
            # Handle case where not all classes are present in y_true
            present_classes = np.unique(y_label_array)
            if len(present_classes) < 2:
                auroc = 0.0
            elif len(present_classes) < self.n_class:
                # Only use columns for classes that exist in y_true
                auroc = roc_auc_score(y_label_array, y_pred_array[:, present_classes],
                                      multi_class='ovr', average='weighted',
                                      labels=present_classes)
            else:
                auroc = roc_auc_score(y_label_array, y_pred_array, multi_class='ovr', average='weighted')
            y_pred_class = np.argmax(y_pred_array, axis=1)
            f1_weighted = sk_f1_score(y_label_array, y_pred_class, average='weighted')
            accuracy = float((y_pred_class == y_label_array).mean())

            if dataloader == "test":
                cm = confusion_matrix(y_label_array, y_pred_class, labels=list(range(self.n_class)))
                per_class_recall = []
                for c in range(self.n_class):
                    if cm[c].sum() > 0:
                        per_class_recall.append(float(cm[c, c] / cm[c].sum()))
                    else:
                        per_class_recall.append(0.0)
                print(f"   Confusion Matrix ({self.n_class} classes):\n{cm}")
                print(f"   Per-class Recall: {[f'{r:.4f}' for r in per_class_recall]}")
                if self.use_multitask and r2_score is not None:
                    print(f"   Regression R²: {r2_score:.4f}")
                    return auroc, f1_weighted, accuracy, test_loss, per_class_recall, r2_score
                else:
                    return auroc, f1_weighted, accuracy, test_loss, per_class_recall
            else:
                if self.use_multitask and r2_score is not None:
                    return auroc, f1_weighted, test_loss, r2_score
                else:
                    return auroc, f1_weighted, test_loss

        # ===== Binary metrics (original) =====
        auroc = roc_auc_score(y_label, y_pred)
        auprc = average_precision_score(y_label, y_pred)

        if dataloader == "test":
            fpr, tpr, thresholds = roc_curve(y_label, y_pred)
            prec, recall, _ = precision_recall_curve(y_label, y_pred)
            precision = tpr / (tpr + fpr)
            f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
            thred_optim = thresholds[5:][np.argmax(f1[5:])]
            y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
            cm1 = confusion_matrix(y_label, y_pred_s)
            accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
            sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
            specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
            if self.experiment:
                self.experiment.log_curve("test_roc curve", fpr, tpr)
                self.experiment.log_curve("test_pr curve", recall, prec)
            precision1 = precision_score(y_label, y_pred_s)

            if self.use_multitask and r2_score is not None:
                print(f"   Regression R²: {r2_score:.4f}")
                return auroc, auprc, np.max(f1[5:]), sensitivity, specificity, accuracy, test_loss, thred_optim, precision1, r2_score
            else:
                return auroc, auprc, np.max(f1[5:]), sensitivity, specificity, accuracy, test_loss, thred_optim, precision1
        else:
            if self.use_multitask and r2_score is not None:
                return auroc, auprc, test_loss, r2_score
            else:
                return auroc, auprc, test_loss
