import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
try:
    train_df = pd.read_csv("train_set_updated_kp_conf.csv")
    val_df = pd.read_csv("test_set_updated_kp_conf.csv")
    test_df = pd.read_csv("val_set_updated_kp_conf.csv")

    # Add a 'set' column to identify each dataset
    train_df['set'] = 'train'
    val_df['set'] = 'validation'
    test_df['set'] = 'test'

    # Combine all data into a single DataFrame
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # --- Create the Grouped Bar Chart ---

    # 1. Define the plotting order for classes and sets
    class_order = combined_df['label_str'].value_counts().index
    hue_order = ['train', 'validation', 'test']
    
    # 2. Calculate percentages for the labels
    crosstab_perc = pd.crosstab(index=combined_df['label_str'], columns=combined_df['set'], normalize='index') * 100
    # Ensure the percentage dataframe is sorted in the same order as the plot will be
    crosstab_perc = crosstab_perc.loc[class_order][hue_order]

    # 3. Create the plot
    fig, ax = plt.subplots(figsize=(20, 12)) # Increased figure size for better spacing
    light_palette = ['mediumpurple', 'red', 'lawngreen']

    sns.countplot(
        data=combined_df,
        x='label_str',
        hue='set',
        order=class_order,
        hue_order=hue_order,
        palette=light_palette,
        dodge=True,  # This is the key change for grouped bars
        ax=ax
    )

    # 4. Add the custom percentage labels to the count plot
    for i, container in enumerate(ax.containers):
        # Get the correct column of percentages for the current container
        set_name = hue_order[i]
        perc_values_for_set = crosstab_perc[set_name]
        
        # Create labels formatted as percentages
        labels = [f'{p:.1f}%' for p in perc_values_for_set]
        
        ax.bar_label(
            container,
            labels=labels,
            label_type='edge',
            fontsize=5,
            padding=5
        )

    # 4. Customize the plot
    ax.set_title('Distribution of Yoga Classes Across Sets', fontsize=20, weight='bold')
    ax.set_xlabel('Number of Images (Count)', fontsize=14)
    ax.set_ylabel('Yoga Pose', fontsize=14)
    ax.legend(title='Set', loc='upper right')
    
    # Rotate the x-axis tick labels (yoga pose names)
    plt.xticks(rotation=45, ha='right')

    # Adjust x-axis limit to give space for labels
    ax.margins(y=0.1) 
    
    ax.spines[['right', 'top']].set_visible(False)
    plt.tight_layout()

    # Save the figure
    plt.savefig("yoga_class_distribution.png", dpi=300)
    print("Grouped distribution plot saved as yoga_class_distribution.png")

except FileNotFoundError as e:
    print(f"Error: {e}. Please make sure the CSV files are in the correct directory.")
except KeyError:
    print("Error: A 'label_str' column was not found. Please check your CSV files.")