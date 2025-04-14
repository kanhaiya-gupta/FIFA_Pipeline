import pandas as pd
import matplotlib.pyplot as plt

def generate_cluster_plot(predictions_path, plot_path):
    """Generate bar plot of cluster distribution."""
    df = pd.read_csv(predictions_path)
    cluster_counts = df['prediction'].value_counts()
    
    plt.figure(figsize=(8, 6))
    plt.bar(cluster_counts.index, cluster_counts.values)
    plt.xlabel("Cluster")
    plt.ylabel("Number of Players")
    plt.title("Player Distribution Across Clusters")
    plt.savefig(plot_path)
    plt.close()
