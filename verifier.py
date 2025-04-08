"""
Verification Module for Counterfeit Drug Detection System

This module provides functions for verifying the authenticity of medicines
by combining barcode extraction and database verification.
"""

import os
import cv2
import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple

# Import local modules
from text_verification.barcode_extraction import process_image_for_barcodes, visualize_results
from text_verification.database import MedicineDatabase


class MedicineVerifier:
    """
    Class for verifying the authenticity of medicines using text-based methods.
    """
    
    def __init__(self, db_path: str = 'medicine_database.db', config_path: Optional[str] = None):
        """
        Initialize the medicine verifier.
        
        Args:
            db_path: Path to the database file
            config_path: Path to configuration file (optional)
        """
        # Default configuration
        self.config = {
            'similarity_threshold': 0.8,
            'confidence_threshold': 0.7,
            'verification_weights': {
                'barcode': 0.6,
                'serial_number': 0.4
            }
        }
        
        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                self.config.update(loaded_config)
        
        # Initialize database
        self.db = MedicineDatabase(db_path)
    
    def verify_medicine(self, image_path: str) -> Dict[str, Any]:
        """
        Verify the authenticity of a medicine from an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing verification results
        """
        # Extract barcodes and serial numbers from the image
        extraction_results = process_image_for_barcodes(image_path)
        
        # Initialize verification results
        verification_results = {
            'is_authentic': False,
            'confidence': 0.0,
            'barcode_verification': None,
            'serial_verification': None,
            'medicine_info': None,
            'extracted_data': extraction_results
        }
        
        # Verify barcodes
        barcode_results = []
        for barcode in extraction_results['barcodes']:
            barcode_data = barcode['data']
            verification = self.db.verify_barcode(barcode_data)
            barcode_results.append({
                'barcode_data': barcode_data,
                'verification': verification
            })
        
        # Verify serial numbers
        serial_results = []
        for serial in extraction_results['serial_numbers']:
            serial_number = serial['text']
            verification = self.db.verify_serial_number(serial_number)
            serial_results.append({
                'serial_number': serial_number,
                'verification': verification
            })
        
        # Add verification results
        verification_results['barcode_verification'] = barcode_results
        verification_results['serial_verification'] = serial_results
        
        # Determine overall authenticity
        if barcode_results or serial_results:
            # Calculate weighted confidence score
            barcode_confidence = 0.0
            if barcode_results:
                # Use the highest confidence from barcode verifications
                barcode_confidence = max(
                    result['verification'].get('confidence', 0.0) 
                    for result in barcode_results
                )
            
            serial_confidence = 0.0
            if serial_results:
                # Use the highest confidence from serial verifications
                serial_confidence = max(
                    result['verification'].get('confidence', 0.0) 
                    for result in serial_results
                )
            
            # Calculate weighted confidence
            weights = self.config['verification_weights']
            if barcode_results and serial_results:
                # If we have both, use weighted average
                overall_confidence = (
                    barcode_confidence * weights['barcode'] +
                    serial_confidence * weights['serial_number']
                )
            elif barcode_results:
                # If we only have barcodes
                overall_confidence = barcode_confidence
            else:
                # If we only have serial numbers
                overall_confidence = serial_confidence
            
            # Determine authenticity based on confidence threshold
            is_authentic = overall_confidence >= self.config['confidence_threshold']
            
            verification_results['is_authentic'] = is_authentic
            verification_results['confidence'] = overall_confidence
            
            # Get medicine information from the most confident verification
            if barcode_confidence >= serial_confidence and barcode_results:
                # Use medicine info from barcode verification
                best_barcode = max(
                    barcode_results, 
                    key=lambda x: x['verification'].get('confidence', 0.0)
                )
                if 'medicine' in best_barcode['verification']:
                    verification_results['medicine_info'] = best_barcode['verification']['medicine']
            elif serial_results:
                # Use medicine info from serial verification
                best_serial = max(
                    serial_results, 
                    key=lambda x: x['verification'].get('confidence', 0.0)
                )
                if 'medicine' in best_serial['verification']:
                    verification_results['medicine_info'] = best_serial['verification']['medicine']
        
        # Log verification attempt
        self._log_verification(verification_results)
        
        return verification_results
    
    def _log_verification(self, verification_results: Dict[str, Any]):
        """
        Log verification attempt to the database.
        
        Args:
            verification_results: Verification results dictionary
        """
        # Extract relevant information for logging
        medicine_id = None
        barcode_data = None
        serial_number = None
        
        if verification_results['medicine_info']:
            medicine_id = verification_results['medicine_info'].get('id')
        
        if verification_results['barcode_verification']:
            barcode_data = verification_results['barcode_verification'][0]['barcode_data']
        
        if verification_results['serial_verification']:
            serial_number = verification_results['serial_verification'][0]['serial_number']
        
        # Create log entry
        log_data = {
            'medicine_id': medicine_id,
            'barcode_data': barcode_data,
            'serial_number': serial_number,
            'is_authentic': verification_results['is_authentic'],
            'confidence': verification_results['confidence'],
            'details': {
                'barcode_count': len(verification_results['barcode_verification']),
                'serial_count': len(verification_results['serial_verification'])
            }
        }
        
        # Log to database
        self.db.log_verification(log_data)
    
    def visualize_verification(self, image_path: str, verification_results: Dict[str, Any], output_path: Optional[str] = None):
        """
        Visualize verification results on the image.
        
        Args:
            image_path: Path to the original image
            verification_results: Verification results from verify_medicine
            output_path: Path to save the visualization (optional)
        """
        # Load the original image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Create a copy for visualization
        vis_image = image.copy()
        
        # Draw barcodes
        for barcode_result in verification_results['barcode_verification']:
            # Find the corresponding barcode in extracted_data
            barcode_data = barcode_result['barcode_data']
            barcode = next(
                (b for b in verification_results['extracted_data']['barcodes'] 
                 if b['data'] == barcode_data),
                None
            )
            
            if barcode:
                # Draw polygon
                points = np.array(barcode['points'])
                
                # Determine color based on verification
                if barcode_result['verification'].get('verified', False):
                    color = (0, 255, 0)  # Green for verified
                else:
                    color = (0, 0, 255)  # Red for unverified
                
                cv2.polylines(vis_image, [points], True, color, 2)
                
                # Draw rectangle
                x, y, w, h = barcode['rect']
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
                
                # Add text
                barcode_type = barcode['type']
                confidence = barcode_result['verification'].get('confidence', 0.0)
                cv2.putText(vis_image, f"{barcode_type}: {confidence:.2f}", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw serial numbers
        for serial_result in verification_results['serial_verification']:
            # Find the corresponding serial in extracted_data
            serial_number = serial_result['serial_number']
            serial = next(
                (s for s in verification_results['extracted_data']['serial_numbers'] 
                 if s['text'] == serial_number),
                None
            )
            
            if serial:
                # Draw rectangle
                x, y, w, h = serial['rect']
                
                # Determine color based on verification
                if serial_result['verification'].get('verified', False):
                    color = (0, 255, 0)  # Green for verified
                else:
                    color = (0, 0, 255)  # Red for unverified
                
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
                
                # Add text
                confidence = serial_result['verification'].get('confidence', 0.0)
                cv2.putText(vis_image, f"SN: {confidence:.2f}", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add overall result
        result_text = "AUTHENTIC" if verification_results['is_authentic'] else "COUNTERFEIT"
        confidence = verification_results['confidence']
        cv2.putText(vis_image, f"Result: {result_text} ({confidence:.2f})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                   (0, 255, 0) if verification_results['is_authentic'] else (0, 0, 255), 
                   2)
        
        # Add medicine info if available
        if verification_results['medicine_info']:
            medicine = verification_results['medicine_info']
            info_text = f"{medicine.get('name', 'Unknown')} - {medicine.get('manufacturer', 'Unknown')}"
            cv2.putText(vis_image, info_text, 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Save or display the visualization
        if output_path:
            cv2.imwrite(output_path, vis_image)
            print(f"Visualization saved to {output_path}")
        else:
            # Convert to RGB for display
            vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
            
            # Display using matplotlib
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 8))
            plt.imshow(vis_image_rgb)
            plt.axis('off')
            plt.title('Medicine Verification Results')
            plt.show()
    
    def close(self):
        """
        Close the database connection.
        """
        self.db.close()


def verify_medicine_from_image(image_path: str, db_path: str = 'medicine_database.db', 
                              output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to verify a medicine from an image.
    
    Args:
        image_path: Path to the image file
        db_path: Path to the database file
        output_path: Path to save the visualization (optional)
        
    Returns:
        Dictionary containing verification results
    """
    verifier = MedicineVerifier(db_path)
    
    try:
        # Verify medicine
        results = verifier.verify_medicine(image_path)
        
        # Visualize results if output path is provided
        if output_path:
            verifier.visualize_verification(image_path, results, output_path)
        
        return results
    
    finally:
        verifier.close()


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Medicine Verification')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--db', type=str, default='medicine_database.db', help='Path to database file')
    parser.add_argument('--output', type=str, help='Path to save visualization')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    
    args = parser.parse_args()
    
    try:
        # Create verifier
        verifier = MedicineVerifier(args.db, args.config)
        
        # Verify medicine
        results = verifier.verify_medicine(args.image)
        
        # Print results
        print(f"Verification Result: {'AUTHENTIC' if results['is_authentic'] else 'COUNTERFEIT'}")
        print(f"Confidence: {results['confidence']:.2f}")
        
        if results['medicine_info']:
            medicine = results['medicine_info']
            print(f"Medicine: {medicine.get('name', 'Unknown')} ({medicine.get('manufacturer', 'Unknown')})")
            print(f"NDC: {medicine.get('ndc', 'Unknown')}")
            print(f"Description: {medicine.get('description', 'Unknown')}")
        
        print(f"\nExtracted {len(results['barcode_verification'])} barcodes and {len(results['serial_verification'])} serial numbers")
        
        # Visualize results
        if args.output:
            verifier.visualize_verification(args.image, results, args.output)
            print(f"Visualization saved to {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Close verifier
        verifier.close()
