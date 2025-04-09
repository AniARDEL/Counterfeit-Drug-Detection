import os
import sys
from drug_detection_model import DrugDetectionModel, GradioInterface

def run_gradio_interface():
    """Run only the Gradio interface with the existing trained model"""
    try:
        # Set up paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, "drug_detection_output", "model")
        
        # Try loading different model files in order of preference
        model_files = [
            os.path.join(model_dir, 'best_finetuned_model.h5'),
            os.path.join(model_dir, 'final_finetuned_model.h5'),
            os.path.join(model_dir, 'best_model.h5'),
            os.path.join(model_dir, 'final_model.h5')
        ]
        
        # Try to load each model in order
        drug_model = None
        for model_path in model_files:
            if os.path.exists(model_path):
                print(f"Loading model from {model_path}...")
                try:
                    drug_model = DrugDetectionModel(model_path=model_path)
                    print(f"Successfully loaded model from {model_path}")
                    break
                except Exception as e:
                    print(f"Error loading model from {model_path}: {str(e)}")
        
        # Check if a model was loaded
        if drug_model is None:
            print("Error: No model could be loaded. Please check the model files.")
            return
        
        # Launch Gradio interface
        print("Launching Gradio interface...")
        interface = GradioInterface(drug_model)
        interface.launch()
    
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_gradio_interface() 