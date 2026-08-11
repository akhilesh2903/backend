"""
Inference Pipeline Module
==========================
Simplified inference module that wraps CNN predictions
and formats output for downstream processing.
"""

from .cnn_model import FetalUltrasoundCNN
import torch


class InferencePipeline:
    """
    High-level inference pipeline combining CNN prediction and feature extraction.
    Handles image preprocessing and result formatting.
    """
    
    def __init__(self, model_name='resnet50', device='cpu'):
        """
        Initialize inference pipeline.
        
        Args:
            model_name (str): Name of CNN model to use
            device (str): Computation device ('cpu' or 'cuda')
        """
        self.model = FetalUltrasoundCNN(model_name=model_name, device=device)
        self.device = device
    
    def run_inference(self, image_path):
        """
        Run complete inference pipeline on ultrasound image.
        
        Args:
            image_path (str): Path to ultrasound image
            
        Returns:
            dict: Formatted inference results
        """
        # Get CNN prediction and embedding
        prediction = self.model.predict(image_path)
        
        if not prediction['success']:
            return {
                'status': 'error',
                'message': prediction['error']
            }
        
        # Format results for downstream modules
        result = {
            'status': 'success',
            'image_path': image_path,
            'cnn_prediction': {
                'predicted_class': prediction['predicted_class'],
                'class_label': prediction['class_label'],
                'confidence_percentage': round(prediction['confidence'], 2),
                'all_probabilities': {
                    k: round(v, 2) for k, v in prediction['all_probabilities'].items()
                },
                'condition_description': prediction['condition_description']
            },
            'image_embedding': prediction['embedding'].tolist(),  # Convert to list for JSON serialization
            'model_info': self.model.get_model_info()
        }
        
        return result
    
    def batch_inference(self, image_paths):
        """
        Run inference on multiple images.
        
        Args:
            image_paths (list): List of image paths
            
        Returns:
            list: List of inference results
        """
        results = []
        for img_path in image_paths:
            result = self.run_inference(img_path)
            results.append(result)
        
        return results
