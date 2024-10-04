import json
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools

def load_training_history(history_path):
    with open(history_path, 'r') as f:
        history = json.load(f)
    return history

def downsample_or_average(data, target_len):
    """ Downsample or average data to match target_len. """
    factor = len(data) // target_len
    if factor > 1:
        # Group and average
        data = [np.mean(data[i:i+factor]) for i in range(0, len(data), factor)]
    return data[:target_len]  # Ensure we don't exceed target_len

def smooth_curve(points, factor=0.9):
    """ Apply exponential moving average to smooth the curve. """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def flatten_if_needed(data):
    """ Flatten a list of lists if needed. """
    if isinstance(data[0], list):
        return [item for sublist in data for item in sublist]
    return data

def plot_multiple_models(model_histories, model_names, save_path=None, title_suffix=''):
    num_models = len(model_histories)
    
    # Dynamically adjust figure size based on the number of models
    fig_width = 14
    fig_height = 7 + (num_models // 3)  # Increase height based on the number of models

    plt.figure(figsize=(fig_width, fig_height))

    # Plot Loss
    plt.subplot(1, 2, 1)
    model_handles = {}
    colors = itertools.cycle(plt.cm.tab10.colors)  # Use tab10 colormap
    
    for model_name, history in model_histories.items():
        train_loss = flatten_if_needed(history.get('train_loss', []))
        val_loss = flatten_if_needed(history.get('val_loss', []))

        # Align training metrics with validation points
        if len(train_loss) > len(val_loss):
            train_loss = downsample_or_average(train_loss, len(val_loss))

        # Smooth the validation curves
        smoothed_val_loss = smooth_curve(val_loss, factor=0.8)

        # Plot with same color but different line styles
        color = next(colors)
        line_train, = plt.plot(train_loss, color=color, linestyle='-')
        plt.plot(smoothed_val_loss, color=color, linestyle='--')

        # Store only the first handle for the model name
        model_handles[model_name] = line_train

    plt.xlabel('Validation Points')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss {title_suffix}', fontsize=16)
    plt.grid(True, linestyle=':')

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    colors = itertools.cycle(plt.cm.tab10.colors)  # Reset color cycle
    for model_name, history in model_histories.items():
        train_accuracy = flatten_if_needed(history.get('train_accuracy', []))
        val_accuracy = flatten_if_needed(history.get('val_accuracy', []))

        # Align training metrics with validation points
        if len(train_accuracy) > len(val_accuracy):
            train_accuracy = downsample_or_average(train_accuracy, len(val_accuracy))

        # Smooth the validation curves
        smoothed_val_accuracy = smooth_curve(val_accuracy, factor=0.8)

        # Plot with same color but different line styles
        color = next(colors)
        plt.plot(train_accuracy, color=color, linestyle='-')
        plt.plot(smoothed_val_accuracy, color=color, linestyle='--')

    plt.xlabel('Validation Points')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Training and Validation Accuracy {title_suffix}', fontsize=16)
    plt.grid(True, linestyle=':')

    # Dynamically adjust the number of columns for the legend based on the number of models
    ncol = min(3, num_models // 2)
    
    # Add the legend below the plots, with enough space for all models
    plt.legend([model_handles[m] for m in model_histories.keys()], 
               [model_names[m] for m in model_histories.keys()],
               loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=ncol)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for the legend
    if save_path:
        plt.savefig(os.path.join(save_path, f'comparison_training_validation_{title_suffix}.png'), dpi=300)
    plt.show()

if __name__ == "__main__":
    # Dictionary of model names and their display names for the legend
    model_configurations = {
        # "AdvancedConv2D__b32_lr5_e100_Adam_MERGE_improve_resplit" : "AConv2D_b32_lr5_e100_Adam",
        # "AdvancedConv2D__b64_lr4_e100_Adam_MERGE_improve_resplit": "AConv2D_b64_lr4_e100_Adam",
        # "SingleStream3D_b64_lr4_e100_Adam_MERGE_improve_resplit" : "SS3D_b64_lr4_e100_Adam",
        # "SingleStream3D_b64_lr5_e100_Adam_MERGE_improve_resplit" : "SS3D_b64_lr5_e100_Adam",
        # "AdvancedConv2DModel1_b64_lr4_e100_Adam_MERGE_improve_resplit": "AConv2DV1_b64_lr4_e100_Adam",
        # "AdvancedConv2DModel1_b64_lr5_e100_Adam_MERGE_improve_resplit" : "AConv2DV1_b64_lr5_e100_Adam",

        #==========================================================================================================================================
        #"AdvancedConv2DModelV2_b64_lr2_e100_Adam_MERGE_improve_resplit" : "AConv2DV2_b64_lr2_e100_Adam",
        #"AdvancedConv2DModelV2_b64_lr3_e100_Adam_MERGE_improve_resplit": "AConv2DV2_b64_lr3_e100_Adam",
        #"AdvancedConv2DModelV2_b64_lr4_e100_Adam_MERGE_improve_resplit" : "AConv2DV2_b64_lr4_e100_Adam",
        #"AdvancedConv2DModelV2_b64_lr5_e100_Adam_MERGE_improve_resplit": "AConv2DV2_b64_lr5_e100_Adam",

        #"AdvancedConv2DModelV3_b64_lr3_e100_Adam_MERGE_improve_resplit" : "AConv2DV3_b64_lr3_e100_Adam",
        #"AdvancedConv2DModelV3_b64_lr4_e100_Adam_MERGE_improve_resplit" : "AConv2DV3_b64_lr4_e100_Adam"

        #==========================================================================================================================================
        # "VerticalStackConv2DModel_b64_lr2_e100_Adam_MERGE_improve_resplit": "VTConv2D_b64_lr2_e100_Adam",
        # "VerticalStackConv2DModel_b64_lr3_e100_Adam_MERGE_improve_resplit": "VTConv2D_b64_lr3_e100_Adam",
        # "VerticalStackConv2DModel_b64_lr4_e100_Adam_MERGE_improve_resplit": "VTConv2D_b64_lr4_e100_Adam",
        # "VerticalStackConv2DModel_b64_lr5_e100_Adam_MERGE_improve_resplit": "VTConv2D_b64_lr5_e100_Adam"

        # "VerticalStackConv2DModel_b64_lr2_e100_Adam_Normal_Scheduler" : "NormScheduler_VTC2_lr2_Adam",
        # "VerticalStackConv2DModel_b64_lr2_e100_SGD_Normal_Scheduler": "NormScheduler_VTC2_lr2_SGD",
        # "VerticalStackConv2DModel_b64_lr3_e100_Adam_Normal_Scheduler": "NormScheduler_VTC2_lr3_Adam",
        # "VerticalStackConv2DModel_b64_lr3_e100_SGD_Normal_Scheduler" : "NormScheduler_VTC2_lr3_SGD",
        # "VerticalStackConv2DModel_b64_lr4_e100_Adam_Normal_Scheduler": "NormScheduler_VTC2_lr4_Adam",
        # "VerticalStackConv2DModel_b64_lr4_e100_SGD_Normal_Scheduler": "NormScheduler_VTC2_lr4_SGD",
        # "VerticalStackConv2DModel_b64_lr5_e100_Adam_Normal_Scheduler": "NormScheduler_VTC2_lr5_Adam",
        # "VerticalStackConv2DModel_b64_lr5_e100_SGD_Normal_Scheduler":"NormScheduler_VTC2_lr5_SGD"

        # "CustomCNN4_dropout_multi_b64_lr2_e100_Adam_Cosine_Scheduler" : "CustomCNN4_dropout_multi_b64_lr2_e100_Adam_Cosine_Scheduler",
        # "CustomCNN4_dropout_multi_b64_lr3_e100_Adam_Cosine_Scheduler" : "CustomCNN4_dropout_multi_b64_lr3_e100_Adam_Cosine_Scheduler",
        # "CustomCNN4_dropout_multi_b64_lr4_e100_Adam_Cosine_Scheduler" : "CustomCNN4_dropout_multi_b64_lr4_e100_Adam_Cosine_Scheduler",
        # "CustomCNN4_dropout_multi_b64_lr5_e100_Adam_Cosine_Scheduler" : "CustomCNN4_dropout_multi_b64_lr5_e100_Adam_Cosine_Scheduler"

        # "ResNetSeparable_b64_lr2_e100_Adam_Cosine_Scheduler" : "ResSep_b64_lr2_AdamCos",
        # "ResNetSeparable_b64_lr3_e100_Adam_Cosine_Scheduler" : "ResSep_b64_lr3_AdamCos",
        # "ResNetSeparable_b64_lr4_e100_Adam_Cosine_Scheduler" : "ResSep_b64_lr4_AdamCos",
        # "ResNetSeparable_b64_lr5_e100_Adam_Cosine_Scheduler" : "ResSep_b64_lr2_AdamCos",
        # "ResNetSeparableCrossAttention_b64_lr3_e100_Adam_Cosine_Scheduler" : "ResSepCrossAtt_b64_lr3_AdamCos",
        # "ResNetSeparableCrossAttention_b64_lr4_e100_Adam_Cosine_Scheduler" : "ResSepCrossAtt_b64_lr4_AdamCos",
        # "ResNetSeparableCrossAttention_b64_lr5_e100_Adam_Cosine_Scheduler" : "ResSepCrossAtt_b64_lr5_AdamCos",
    
        # "EnhancedCrossAttentionClass_b64_lr2_e100_Adam_Cosine_Scheduler" : "EnhancCrossAtt_lr2",
        # "EnhancedCrossAttentionClass_b64_lr3_e100_Adam_Cosine_Scheduler" : "EnhancCrossAtt_lr3",
        # "EnhancedCrossAttentionClass_b64_lr4_e100_Adam_Cosine_Scheduler" : "EnhancCrossAtt_lr4",
        # "EnhancedCrossAttentionClass_b64_lr5_e100_Adam_Cosine_Scheduler" : "EnhancCrossAtt_lr5"
    
        # "ResNetSeparableOptimized__b64_lr2_e100_Adam_Cosine_Scheduler" : "ResSepOptimized_lr2",
        # "ResNetSeparableOptimized__b64_lr3_e100_Adam_Cosine_Scheduler" : "ResSepOptimized_lr3",
        # "ResNetSeparableOptimized__b64_lr4_e100_Adam_Cosine_Scheduler" : "ResSepOptimized_lr4",
        # "ResNetSeparableOptimized__b64_lr5_e100_Adam_Cosine_Scheduler" : "ResSepOptimized_lr5",

        "ResNet18Separable_b64_lr2_e100_Adam_StepLR" : "R18_lr2_Adam",
        "ResNet18Separable_b64_lr3_e100_Adam_StepLR" : "R18_lr3_Adam",
        "ResNet18Separable_b64_lr4_e100_Adam_StepLR" : "R18_lr4_Adam",
        "ResNet18Separable_b64_lr5_e100_Adam_StepLR" : "R18_lr5_Adam",

        "ResNet18Separable_b64_lr2_e100_SGD_StepLR" : "R18_lr2_SGD",
        "ResNet18Separable_b64_lr3_e100_SGD_StepLR" : "R18_lr3_SGD",
        "ResNet18Separable_b64_lr4_e100_SGD_StepLR" : "R18_lr4_SGD",
        "ResNet18Separable_b64_lr5_e100_SGD_StepLR" : "R18_lr5_SGD"


    }

    model_histories = {}

    # Load the history for each model
    for model_architecture, display_name in model_configurations.items():
        history_dir = os.path.join('models', model_architecture, 'history')
        history_epochs_file = os.path.join(history_dir, 'training_history_epochs.json')
        
        if os.path.exists(history_epochs_file):
            model_histories[model_architecture] = load_training_history(history_epochs_file)
        else:
            print(f"History file not found for model {model_architecture}")

    # Plot comparison of all models
    # plot_multiple_models(model_histories, model_configurations, save_path="/home/h3trinh/MVMS_HAR", title_suffix='VERTICAL_STACK')
    # plot_multiple_models(model_histories, model_configurations, save_path="/home/h3trinh/MVMS_HAR", title_suffix='AConv2_3DV0_1')
    # plot_multiple_models(model_histories, model_configurations, save_path="/home/h3trinh/MVMS_HAR", title_suffix='NormScheduler_AConv2D')
    # plot_multiple_models(model_histories, model_configurations, save_path="/home/h3trinh/MVMS_HAR", title_suffix='ResNetSeparable_ACosine')
    # plot_multiple_models(model_histories, model_configurations, save_path="/home/h3trinh/MVMS_HAR", title_suffix='EnhancedCrossAttentnion_AdamWRestart_ConstrativeLoss')
    # plot_multiple_models(model_histories, model_configurations, save_path="/home/h3trinh/MVMS_HAR", title_suffix='ResNetSeparableOptimized_AdamCosine')
    plot_multiple_models(model_histories, model_configurations, save_path="/home/h3trinh/MVMS_HAR", title_suffix='ResNetSeparable18')
