import os
from datetime import datetime
import argparse
import tensorflow as tf

from src.split_dataset   import split_dataset
from src.dataloader      import get_binary_pipelines
from src.model           import build_model
from src.train           import train_model
from src.utils           import plot_history, plot_metrics, create_run_directories, plot_dataset_distribution
from src.evaluate        import evaluate_model
from src.gradcam_analysis import analyze_tf_keras_gradcam
from src.run_manager     import RunManager
from src.callbacks       import get_training_callbacks
from src.cross_validation import cross_validate_model
import splitfolders

if __name__ == "__main__":
    print("?? Starting Drowsy Driver Detection Project...")
    print("=" * 50)
    
    # CLI args
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cv", action="store_true", help="Run k-fold cross-validation before standard training")
    parser.add_argument("--cv_folds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--cv_epochs", type=int, default=30, help="Number of epochs per fold during cross-validation")
    parser.add_argument("--cv_base_dir", type=str, default="ydd_splitted_dataset_cv",
                        help="Base directory for CV dataset (e.g. ydd_splitted_dataset_cv)")
    args = parser.parse_args()
    
    # Project root directory: the folder where this file is located
    import os
    project_root = os.path.dirname(os.path.abspath(__file__))
    #project_root = r"D:\internship\Drowsy-Driver-Detection-Project"

    # 1) Raw data folder (what you have)
    raw_dir = os.path.join(project_root, "yawning_detection_dataset")
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"`dataset` not found: {raw_dir}")

    # 2) Folder where split data will go (main training)
    output_dir = os.path.join(project_root, "ydd_splitted_dataset")
    print(f"?? Data directory: {output_dir}")

    # 3) Create Train/Val/Test folder hierarchy
    #    raw_dir contains => drowsy, notdrowsy
    #split_dataset(raw_dir, output_dir, classes=("NoYawn", "Yawn"), seed=42)
    #splitfolders.ratio(raw_dir, output=output_dir, seed=1337, ratio=(0.7, 0.15, 0.15))
    
    # 4) EXPERIMENT CONFIGURATION
    # ============================================================
    GRADCAM_WEIGHTS_FILE = os.path.join(project_root, "artifacts", "reward-landmark-soft","exp_weights.json")
    #exp_weights optimized_gradcam_weights
    
    # Create run name based on configuration
    run_name = "30_epoch_exp-reward-soft-sw-yawning"
    
    # 5) Create run manager
    print("📁 Creating run manager...")
    run_manager = RunManager(run_name)
    print(f"✅ Run manager created: {run_manager.run_dir}")
    print(tf.__version__); print(tf.config.list_physical_devices('GPU'))

    # Optional: run k-fold cross-validation on a (possibly different) train split
    if args.use_cv:
        cv_base_dir = os.path.join(project_root, args.cv_base_dir)
        print(f"?? CV data directory: {cv_base_dir}")
        print("?? Running cross-validation...")
        cv_results = cross_validate_model(
            base_dir=cv_base_dir,
            k=args.cv_folds,
            img_size=(224, 224),
            batch_size=32,
            epochs=args.cv_epochs,
            class_names=("NoYawn", "Yawn"),
        )
        print(f"CV results: {cv_results}")

        # Save a simple text summary under the current run directory
        cv_summary_path = os.path.join(run_manager.run_dir, "cv_summary.txt")
        with open(cv_summary_path, "w", encoding="utf-8") as f:
            f.write("Cross-validation summary\n")
            f.write(f"Folds       : {args.cv_folds}\n")
            f.write(f"Epochs/fold : {args.cv_epochs}\n")
            f.write(f"Base dir    : {cv_base_dir}\n\n")
            f.write(f"val_accuracy_mean = {cv_results.get('val_accuracy_mean', float('nan')):.4f}\n")
            f.write(f"val_accuracy_std  = {cv_results.get('val_accuracy_std', float('nan')):.4f}\n")
            f.write(f"val_auc_mean      = {cv_results.get('val_auc_mean', float('nan')):.4f}\n")
            f.write(f"val_auc_std       = {cv_results.get('val_auc_std', float('nan')):.4f}\n")
        print(f"?? CV summary saved to: {cv_summary_path}")

    # 6) tf.data pipelines
    # LOADER (binary: NotDrowsy=0, Drowsy=1)
    print("🔄 Loading datasets...")
    
    train_ds, val_ds, test_ds, class_names = get_binary_pipelines(
        output_dir,
        img_size=(224, 224),
        batch_size=32,
        seed=42,
        gradcam_weights_path=GRADCAM_WEIGHTS_FILE
    )
    
    # Print configuration
    print("\n" + "=" * 50)
    print("📋 EXPERIMENT CONFIGURATION:")
    print("=" * 50)
    print("=" * 50 + "\n")
    
    print("✅ Datasets loaded successfully!")
    '''
    # Debug dataset shapes (support (x,y) and (x,y,w))
    def _print_batch_info(ds, name):
        for batch in ds.take(2):
            if isinstance(batch, (tuple, list)) and len(batch) == 3:
                x_batch, y_batch, w_batch = batch
                print(f"{name} x:", x_batch.shape, " y:", y_batch.shape, " w:", w_batch.shape)
                print(tf.reduce_mean(y_batch), tf.reduce_mean(w_batch))
            else:
                x_batch, y_batch = batch
                print(f"{name} x:", x_batch.shape, " y:", y_batch.shape)
                print(tf.reduce_mean(y_batch))

    _print_batch_info(train_ds, "train")
    '''
    # 5.1) Plot dataset distribution
    print("?? Analyzing dataset distribution...")
    dist_plot_path = os.path.join(run_manager.run_dir, "plots", "dataset_distribution.png")
    # distribution plot
    plot_dataset_distribution(output_dir, save_path=dist_plot_path)
    print("? Dataset distribution analyzed and saved!")

    # 6) Build and train model
    print("???  Building model...")
    model = build_model()
    print("? Model built successfully!")
    
    # 6.1) Check for existing checkpoint and load if available
    print("?? Checking for existing checkpoints...")
    initial_epoch = run_manager.load_latest_checkpoint(model)
    
    if initial_epoch > 0:
        print(f"?? Resuming training from epoch {initial_epoch + 1}")
    else:
        print("?? Starting training from scratch")
    
    # 7) Save initial config
    epoch_count=30
    config = {
        "run_name": run_manager.run_name,
        "epochs": epoch_count,
        "input_shape": (224, 224, 3),
        "model_type": "CNN",
        "classes": list(class_names),
        "batch_size": 32,
        "learning_rate": 1e-3,
        "started_at": str(datetime.now()),
        "initial_epoch": initial_epoch
    }
    run_manager.save_config(config)
    
    # 8) Training with all callbacks
    print("?? Starting training...")
    
    # Get all training callbacks (custom + standard Keras callbacks)
    gradcam_epoch_outputs = os.path.join(run_manager.run_dir, "gradcam_epoch_outputs")
    gradcam_log_file = os.path.join(run_manager.run_dir, "gradcam_debug.log")
    sample_weight_log_file = os.path.join(run_manager.run_dir, "sample_weights_stats.json")
    callbacks = get_training_callbacks(
        run_manager, 
        val_ds, 
        gradcam_epoch_outputs, 
        max_samples=10, 
        gradcam_log_file=gradcam_log_file,
        train_ds=train_ds,  # Pass train_ds for sample_weight monitoring
        monitor_sample_weights=True,  # Enable sample_weight monitoring
        sample_weight_log_file=sample_weight_log_file
    )
    
    # Train the model
    history = train_model(
        model, 
        train_ds, 
        val_ds, 
        epochs=epoch_count,
        callbacks=callbacks,  # Add all callbacks
        initial_epoch=initial_epoch,  # Resume from checkpoint if available
    )
    print("? Training completed!")

    # 9) Plot training graphs and save them
    print("?? Plotting training history...")
    history_plot_path = os.path.join(run_manager.run_dir, "plots", "training_history.png")
    plot_history(history, save_path=history_plot_path)
    
    print("?? Plotting metrics...")
    metrics_plot_path = os.path.join(run_manager.run_dir, "plots", "training_metrics.png")
    plot_metrics(history, save_path=metrics_plot_path)

    
    # 9.5) Evaluate on training set
    print("?? Evaluating model on training set...")
    train_plots_dir = os.path.join(run_manager.run_dir, "plots", "train_gradcam")
    os.makedirs(train_plots_dir, exist_ok=True)
    analyze_tf_keras_gradcam(
        model=model,
        test_ds=train_ds,
        output_dir=train_plots_dir,
        num_samples=30,
        class_names=tuple(class_names)
    )
    print("? Training evaluation completed!")

    # 10) Evaluate on validation set
    print("?? Evaluating model on validation set...")
    evaluate_model(
        model,
        val_ds,
        plots_dir=os.path.join(run_manager.run_dir, "plots"),
        subject_diverse_dir=os.path.join(output_dir, "val"),
        ds_name="val",
    )
    print("? Validation evaluation completed!")
    
    # 11) Evaluate on test set
    print("?? Evaluating model on test set...")
    evaluate_model(
        model,
        test_ds,
        plots_dir=os.path.join(run_manager.run_dir, "plots"),
        subject_diverse_dir=os.path.join(output_dir, "test"),
        ds_name="test",
    )
    print("? Test evaluation completed!")

    # 11) Save final model
    print("?? Saving final model...")
    run_manager.save_final_model(model)
    
    # 12) Save simple config
    config = {
        "run_name": run_manager.run_name,
        "epochs": epoch_count,
        "input_shape": (224, 224, 3),
        "model_type": "CNN",
        "classes": list(class_names)
    }
    
    import json
    config_path = os.path.join(run_manager.run_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"? Config saved to: {config_path}")
    
    print("\n" + "=" * 50)
    print("?? All tasks completed successfully!")
    print(f"?? Results saved to: {run_manager.run_dir}")
    print("Project finished! ??")