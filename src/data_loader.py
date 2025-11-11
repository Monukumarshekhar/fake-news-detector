import pandas as pd
import os

def load_data():
    # Define paths to the datasets
    base_path = os.path.dirname(os.path.dirname(__file__))
    fake_path = os.path.join(base_path, 'data', 'Fake.csv')
    true_path = os.path.join(base_path, 'data', 'True.csv')

    print("Loading datasets...")
    # Read the CSV files
    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)

    # Add a 'label' column: 0 for fake, 1 for true
    df_fake['label'] = 0
    df_true['label'] = 1

    # Combine them into one dataframe
    df = pd.concat([df_fake, df_true], axis=0).reset_index(drop=True)
    
    # We only need the 'text' and the 'label' for this project
    # (You can also use 'title' if you want, but let's start simple)
    df = df[['text', 'label']]

    # Shuffle the dataset randomly
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Data loaded successfully! Total records: {len(df)}")
    print(f"Fake news count: {len(df[df['label'] == 0])}")
    print(f"Real news count: {len(df[df['label'] == 1])}")
    
    return df

if __name__ == "__main__":
    # This block only runs if you execute this script directly
    df = load_data()
    print("\nSample data:")
    print(df.head())
    