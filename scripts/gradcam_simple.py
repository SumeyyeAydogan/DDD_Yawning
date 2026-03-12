"""
Simple script to test GradCAM visualization.
Modify the variables below to test different models and settings.
"""

import os
import sys
import tensorflow as tf

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.gradcam_analysis import analyze_subjects_gradcam, analyze_tf_keras_gradcam
from src.dataloader import get_binary_pipelines
from src.losses import create_simple_masked_loss

# ============================================
# CONFIGURATION - Modify these as needed
# ============================================

# Model path
MODEL_PATH = os.path.join(project_root, "runs", "30_epoch_without-mask_sbj-gradcam-fixed", "models", "final_model.h5")
#"30_epoch_without-mask_sbj-gradcam-fixed"
# Dataset directory
DATASET_ROOT = os.path.join(project_root, "splitted_dataset")

# Output directory (if None, will be auto-generated next to model)
OUTPUT_DIR = None

# Number of samples to process
NUM_SAMPLES = 54

# Method: "custom" or "tf_keras_vis"
METHOD = "custom" #"custom" "tf_keras_vis"

# Random seed
SEED = 42

# Save only misclassified samples (FP/FN) if True
ONLY_FALSE = False

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("🎯 GradCAM Test Script")
    print("=" * 50)
    
    # Load model
    print(f"📂 Loading model: {MODEL_PATH}")
    try:
        # Create custom loss function for loading
        loss_fn = create_simple_masked_loss()
        
        # Load model with custom objects
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={'loss_fn': loss_fn}
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Determine output directory
    if OUTPUT_DIR is None:
        model_dir = os.path.dirname(MODEL_PATH)
        if os.path.basename(model_dir) == "models":
            run_root = os.path.dirname(model_dir)
        else:
            run_root = model_dir
        
        method_suffix = "tf_keras" if METHOD == "tf_keras_vis" else "custom"
        OUTPUT_DIR = os.path.join(run_root, f"gradcam_{method_suffix}")
        if ONLY_FALSE:
            OUTPUT_DIR = f"{OUTPUT_DIR}_missclassified"
    print(f"📁 Output directory: {OUTPUT_DIR}")
    
    # Run analysis based on method
    if METHOD == "tf_keras_vis":
        print(f"\n🎯 Running tf-keras-vis GradCAM analysis...")
        try:
            train_ds, val_ds, test_ds, class_names = get_binary_pipelines(DATASET_ROOT)
            print("✅ Dataset loaded from pipeline")
            
            analyze_tf_keras_gradcam(
                model=model,
                test_ds=test_ds,
                output_dir=OUTPUT_DIR,
                num_samples=NUM_SAMPLES,
                class_names=tuple(class_names),
                seed=SEED,
                include_buckets=("FP", "FN") if ONLY_FALSE else None
            )
        except ImportError as e:
            print(f"❌ Error: {e}")
            print("   Install tf-keras-vis with: pip install tf-keras-vis")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            sys.exit(1)
    else:
        print(f"\n🎯 Running custom GradCAM analysis...")
        test_dir = os.path.join(DATASET_ROOT, "test")
        
        if not os.path.exists(test_dir):
            print(f"❌ Test directory not found: {test_dir}")
            sys.exit(1)
        
        try:
            analyze_subjects_gradcam(
                model=model,
                test_dir=test_dir,
                output_dir=OUTPUT_DIR,
                num_samples=NUM_SAMPLES,
                seed=SEED,
                include_buckets=("FP", "FN") if ONLY_FALSE else None
            )
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            sys.exit(1)
    
    print(f"\n✅ Analysis completed!")
    print(f"📁 Results saved to: {OUTPUT_DIR}")