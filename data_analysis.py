    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Performance Metrics vs. Threshold', fontsize=14)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('output/analysis/performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_feature_comparison():
    """
    Create a visual comparison of features between authentic and counterfeit samples.
    """
    print("Analyzing feature comparisons...")
    
    # Create figure for feature comparison
    plt.figure(figsize=(12, 8))
    
    # Define features to compare
    features = ['Color Consistency', 'Texture Uniformity', 'Logo Similarity', 'Print Quality', 'Shape Regularity']
    
    # Simulated scores for authentic and counterfeit samples
    authentic_scores = [0.92, 0.88, 0.95, 0.90, 0.93]
    counterfeit_scores = [0.71, 0.65, 0.58, 0.62, 0.68]
    
    # Set width of bars
    barWidth = 0.3
    
    # Set position of bars on X axis
    r1 = np.arange(len(features))
    r2 = [x + barWidth for x in r1]
    
    # Create bars
    plt.bar(r1, authentic_scores, width=barWidth, edgecolor='white', label='Authentic', color='#2ecc71')
    plt.bar(r2, counterfeit_scores, width=barWidth, edgecolor='white', label='Counterfeit', color='#e74c3c')
    
    # Add labels and legend
    plt.xlabel('Feature', fontweight='bold', fontsize=12)
    plt.ylabel('Score', fontweight='bold', fontsize=12)
    plt.title('Feature Comparison: Authentic vs. Counterfeit', fontsize=16)
    plt.xticks([r + barWidth/2 for r in range(len(features))], features, rotation=45)
    plt.ylim(0, 1)
    
    # Create legend & Show graphic
    plt.legend()
    plt.tight_layout()
    
    plt.savefig('output/analysis/feature_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_comprehensive_analysis_report():
    """
    Create a comprehensive analysis report combining multiple visualizations.
    """
    print("Creating comprehensive analysis report...")
    
    # Create a large figure for the comprehensive report
    plt.figure(figsize=(20, 16))
    
    # Load the individual visualizations
    try:
        dataset_comp = plt.imread('output/analysis/dataset_composition.png')
        feature_dist = plt.imread('output/analysis/feature_distributions.png')
        perf_metrics = plt.imread('output/analysis/performance_metrics.png')
        feature_comp = plt.imread('output/analysis/feature_comparison.png')
        
        # Plot the visualizations in a grid
        plt.subplot(2, 2, 1)
        plt.imshow(dataset_comp)
        plt.title('Dataset Composition', fontsize=16)
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.imshow(feature_dist)
        plt.title('Feature Distributions', fontsize=16)
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.imshow(perf_metrics)
        plt.title('Model Performance Metrics', fontsize=16)
        plt.axis('off')
        
        plt.subplot(2, 2, 4)
        plt.imshow(feature_comp)
        plt.title('Feature Comparison', fontsize=16)
        plt.axis('off')
        
        # Add a title to the entire figure
        plt.suptitle('Comprehensive Analysis of Counterfeit Drug Detection System', fontsize=24)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
        plt.savefig('output/analysis/comprehensive_analysis_report.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Comprehensive analysis report created successfully!")
    except Exception as e:
        print(f"Error creating comprehensive report: {str(e)}")

def main():
    """
    Main function to run the data analysis.
    """
    print("Starting data analysis for counterfeit drug detection...")
    
    # Load and prepare data
    features_df, metadata_df = load_sample_data()
    
    # Perform analyses
    analyze_dataset_composition(metadata_df)
    analyze_feature_distributions(features_df, metadata_df)
    analyze_model_performance()
    analyze_feature_comparison()
    
    # Create comprehensive report
    create_comprehensive_analysis_report()
    
    print("Data analysis completed successfully!")
    print(f"Visualizations saved to: {os.path.abspath('output/analysis/')}")

if __name__ == "__main__":
    main()