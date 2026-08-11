"""
Grad-CAM Visualization Module
==============================
Generates Grad-CAM (Gradient-weighted Class Activation Map) visualizations
to highlight the regions of the ultrasound image that contribute most to
the CNN's classification decision.
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class GradCAM:
    """
    Generates Grad-CAM heatmaps for CNN predictions.
    Helps identify which image regions influenced the model decision.
    """
    
    def __init__(self, model, device='cpu'):
        """
        Initialize Grad-CAM.
        
        Args:
            model: PyTorch model
            device: 'cpu' or 'cuda'
        """
        self.model = model
        self.device = device
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture gradients and activations."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Register hooks on the final convolutional layer
        if hasattr(self.model, 'layer4'):  # ResNet
            self.model.layer4[-1].register_forward_hook(forward_hook)
            self.model.layer4[-1].register_backward_hook(backward_hook)
        else:  # MobileNet or others
            for module in self.model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    last_conv = module
            if hasattr(self, 'last_conv'):
                last_conv.register_forward_hook(forward_hook)
                last_conv.register_backward_hook(backward_hook)
    
    def generate(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor [1, 3, H, W]
            class_idx: Target class index (if None, uses predicted class)
            
        Returns:
            np.ndarray: Grad-CAM heatmap [H, W]
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        
        # Get the score for the target class
        target_score = output[0, class_idx]
        
        # Backward pass
        self.model.zero_grad()
        target_score.backward()
        
        # Compute Grad-CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Compute weights
        weights = gradients.mean(dim=(1, 2))  # [C]
        
        # Compute weighted activation
        cam = torch.zeros_like(activations[0])
        for i, weight in enumerate(weights):
            cam += weight * activations[i]
        
        # Apply ReLU to get only positive contributions
        cam = F.relu(cam)
        
        # Normalize
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        
        return cam.cpu().numpy()
    
    def visualize(self, image_path, output_path=None, heatmap_intensity=0.5):
        """
        Create visualization overlaying Grad-CAM on original image.
        
        Args:
            image_path (str): Path to input ultrasound image
            output_path (str): Path to save visualization (optional)
            heatmap_intensity (float): Intensity of heatmap overlay (0-1)
            
        Returns:
            np.ndarray: Visualization image
        """
        # Load and preprocess image
        from torchvision import transforms
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        input_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Generate Grad-CAM
        cam = self.generate(input_tensor)
        
        # Resize to match original image size
        cam = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]))
        
        # Create heatmap using jet colormap
        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam), 
            cv2.COLORMAP_JET
        )
        
        # Overlay on original image
        visualization = cv2.addWeighted(
            image_np, 
            1 - heatmap_intensity,
            heatmap, 
            heatmap_intensity, 
            0
        )
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        
        return visualization


class ClinicalVisualization:
    """
    Create comprehensive visualizations for clinical reporting.
    """
    
    @staticmethod
    def create_diagnostic_figure(image_path, grad_cam_overlay=None, 
                                 prediction_info=None, save_path=None):
        """
        Create a comprehensive diagnostic figure with multiple visualizations.
        
        Args:
            image_path (str): Path to original image
            grad_cam_overlay (np.ndarray): Grad-CAM visualization
            prediction_info (dict): CNN prediction information
            save_path (str): Path to save figure
            
        Returns:
            matplotlib figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Load original image
        original_image = Image.open(image_path)
        
        # Plot original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Ultrasound Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Plot Grad-CAM overlay if provided
        if grad_cam_overlay is not None:
            axes[1].imshow(cv2.cvtColor(grad_cam_overlay, cv2.COLOR_BGR2RGB))
            axes[1].set_title('Grad-CAM Attention Map\n(Red regions = High Model Focus)', 
                             fontsize=12, fontweight='bold')
        else:
            axes[1].imshow(original_image)
            axes[1].set_title('No Attention Map Available', fontsize=12, fontweight='bold')
        
        axes[1].axis('off')
        
        # Add prediction info as text if provided
        if prediction_info:
            fig.text(0.5, 0.02, 
                    f"Predicted: {prediction_info.get('class_label', 'Unknown')} | "
                    f"Confidence: {prediction_info.get('confidence', 0):.1f}%",
                    ha='center', fontsize=11, style='italic')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def create_confidence_plot(all_probabilities, save_path=None):
        """
        Create bar plot of classification probabilities.
        
        Args:
            all_probabilities (dict): Class probabilities
            save_path (str): Path to save figure
            
        Returns:
            matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        classes = list(all_probabilities.keys())
        probs = list(all_probabilities.values())
        
        colors = ['#2ecc71' if p == max(probs) else '#3498db' for p in probs]
        
        bars = ax.barh(classes, probs, color=colors, edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Confidence (%)', fontsize=11, fontweight='bold')
        ax.set_title('Classification Confidence Distribution', fontsize=13, fontweight='bold')
        ax.set_xlim(0, 100)
        
        # Add percentage labels on bars
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            ax.text(prob + 2, i, f'{prob:.1f}%', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
